//! The general fuse — stages C and D of
//! [`docs/brep-boolean-v2-plan.md`](../../docs/brep-boolean-v2-plan.md).
//!
//! # Why a second boolean
//!
//! [`Body::boolean`] splits one face at a time by the curves computed for that
//! face's pairs, and recovers afterwards which of the pieces' boundaries are
//! the same edge. Everything it gets wrong is that recovery: an edge of one
//! solid lying in a face of the other, an outline that two faces each cut
//! their own way, a split point one face found and its neighbour did not.
//!
//! Here nothing needs recovering, because nothing is computed twice:
//!
//! 1. **One vertex pool.** Every input vertex, every end of a cut, every
//!    crossing goes through one table welded at the tolerance, so a point two
//!    faces both find is one index.
//! 2. **Each pair once.** A face of A and a face of B are intersected once and
//!    the section is handed to both: for two planes the line they share,
//!    otherwise the curves [`ssi`] or [`march`] finds, clipped to the part
//!    lying in both faces. Faces on one surface hand each other their
//!    outlines instead.
//! 3. **An arrangement per face**, in its own parameters. The face's rings
//!    and the sections it was handed are split wherever they cross or touch,
//!    and the faces of that graph are walked. A face that wraps round its
//!    surface is a whole period wide with its seam on both sides, and every
//!    section is laid down once per period it reaches.
//! 4. **Classification** of one point per piece: on a face the other solid
//!    shares, by the rule for shared walls; otherwise by a ray against the
//!    other solid, exactly when its faces are all flat and against its
//!    tessellation when they are not.
//! 5. **Assembly.** Pieces of one wrapping face that meet across its seam are
//!    one face again, and one that meets itself keeps the seam as an edge it
//!    uses twice. Every side is split at every vertex lying on it, after
//!    which an edge is a pair of pool indices and the two faces along it name
//!    the same pair. Runs of those between the same two faces are chained
//!    back into one edge.
//!
//! Kept pieces are not merged otherwise, as OpenCASCADE's general fuse leaves
//! them. Anything the method cannot settle — a face it cannot lay out, a ray
//! that grazes in every direction, an edge with other than two faces — is a
//! [`Declined`], and [`Body::boolean`] falls back to its own path.

use std::collections::{HashMap, HashSet};

use crate::nurbs::v3;
use crate::nurbs::V3;

use super::body::{Body, Edge, Face, TrimLoop};
use super::boolean::{point_in_mesh, BooleanOp, Declined};
use super::curve3d::Curve3d;
use super::intersect::{march, ssi, SsiResult};
use super::Surface;

type P2 = [f64; 2];

/// A ring of a face: its points in space and in the surface's parameters.
type Ring = (Vec<V3>, Vec<(f64, f64)>);

/// A link across a seam: the piece beyond, the direction, which way, and
/// the two sides it joins.
type Link = (usize, usize, i64, SideAt, SideAt);

// -- the plane ---------------------------------------------------------------

fn sub2(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross2(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn dot2(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn len2(a: P2) -> f64 {
    a[0].hypot(a[1])
}

/// Distance from `p` to the segment `ab`, and the parameter of the nearest
/// point, clamped to it.
fn to_segment(p: P2, a: P2, b: P2) -> (f64, f64) {
    let ab = sub2(b, a);
    let l = dot2(ab, ab);
    let t = if l > 0.0 {
        (dot2(sub2(p, a), ab) / l).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let at = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    (len2(sub2(p, at)), t)
}

/// Inside an odd number of the rings.
fn in_rings(rings: &[Vec<P2>], p: P2) -> bool {
    let mut inside = false;
    for r in rings {
        let n = r.len();
        for i in 0..n {
            let (a, b) = (r[i], r[(i + 1) % n]);
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                if p[0] < x {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

fn ring_distance(rings: &[Vec<P2>], p: P2) -> f64 {
    let mut best = f64::INFINITY;
    for r in rings {
        let n = r.len();
        for i in 0..n {
            best = best.min(to_segment(p, r[i], r[(i + 1) % n]).0);
        }
    }
    best
}

fn signed_area(ring: &[P2]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| cross2(ring[i], ring[(i + 1) % n]))
        .sum::<f64>()
        * 0.5
}

/// Distance from `p` to the polyline `line`.
fn to_polyline(p: V3, line: &[V3]) -> f64 {
    line.windows(2)
        .map(|w| {
            let ab = v3::sub(w[1], w[0]);
            let l = v3::dot(ab, ab);
            let t = if l > 0.0 {
                (v3::dot(v3::sub(p, w[0]), ab) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            v3::dist(p, v3::add(w[0], v3::scale(ab, t)))
        })
        .fold(f64::INFINITY, f64::min)
}

fn lerp3(a: V3, b: V3, t: f64) -> V3 {
    v3::add(a, v3::scale(v3::sub(b, a), t))
}

// -- faces as parameter regions ----------------------------------------------

/// A face of an operand, laid out in its surface's parameters.
///
/// Parameters are *scaled* by the surface's speed at the face's middle, so a
/// distance in them is roughly a distance in space and one tolerance serves
/// for both.
struct Patch {
    surface: Surface,
    flipped: bool,
    /// Origin and normal, when the surface is a plane.
    plane: Option<(V3, V3)>,
    scale: P2,
    /// The middle of the face, in the surface's own parameters.
    anchor: (f64, f64),
    /// Periods, scaled.
    period: [Option<f64>; 2],
    /// The domain: each ring's parameters, and the points in space they are.
    rings2: Vec<Vec<P2>>,
    rings3: Vec<Vec<V3>>,
    /// Box of the domain in scaled parameters.
    dlo: P2,
    dhi: P2,
    /// Box in space.
    lo: V3,
    hi: V3,
    /// Whether the domain is a whole period across, in u and in v.
    wraps: [bool; 2],
}

impl Patch {
    fn uv(&self, p: V3) -> Option<P2> {
        let (u, v) = self.surface.invert_near(p, self.anchor)?;
        Some([u * self.scale[0], v * self.scale[1]])
    }

    fn point(&self, w: P2) -> V3 {
        self.surface
            .point(w[0] / self.scale[0], w[1] / self.scale[1])
    }

    /// `q` and the same point a period either way, where the surface has one.
    fn images(&self, q: P2) -> Vec<P2> {
        let shifts = |p: Option<f64>| match p {
            Some(t) => vec![0.0, -t, t],
            None => vec![0.0],
        };
        let mut out = Vec::with_capacity(9);
        for du in shifts(self.period[0]) {
            for dv in shifts(self.period[1]) {
                out.push([q[0] + du, q[1] + dv]);
            }
        }
        out
    }

    /// Off the surface by more than the tolerance?
    fn off_surface(&self, p: V3, tolerance: f64) -> bool {
        match self.plane {
            Some((o, n)) => v3::dot(n, v3::sub(p, o)).abs() > tolerance * 4.0,
            None => self
                .uv(p)
                .is_none_or(|q| v3::dist(self.point(q), p) > tolerance * 4.0),
        }
    }

    /// On the face or its boundary.
    fn holds(&self, p: V3, tolerance: f64) -> bool {
        self.holds_within(p, tolerance, tolerance)
    }

    /// On the face, within `tolerance` of its surface and `reach` of its
    /// boundary.
    fn holds_within(&self, p: V3, tolerance: f64, reach: f64) -> bool {
        if self.off_surface(p, tolerance) {
            return false;
        }
        let Some(q) = self.uv(p) else {
            return false;
        };
        self.images(q)
            .into_iter()
            .any(|s| ring_distance(&self.rings2, s) <= reach || in_rings(&self.rings2, s))
    }

    /// Inside the face, clear of its boundary.
    fn interior(&self, p: V3, tolerance: f64) -> bool {
        if self.off_surface(p, tolerance) {
            return false;
        }
        let Some(q) = self.uv(p) else {
            return false;
        };
        self.images(q)
            .into_iter()
            .any(|s| in_rings(&self.rings2, s) && ring_distance(&self.rings2, s) > tolerance)
    }

    /// A polyline's parameters, continued across the period rather than
    /// jumping at it.
    fn unwrapped(&self, pts: &[V3]) -> Option<Vec<P2>> {
        let mut out: Vec<P2> = Vec::with_capacity(pts.len());
        for p in pts {
            let mut q = self.uv(*p)?;
            if let Some(prev) = out.last() {
                for k in 0..2 {
                    if let Some(t) = self.period[k] {
                        q[k] -= t * ((q[k] - prev[k]) / t).round();
                    }
                }
            }
            out.push(q);
        }
        Some(out)
    }

    /// The shifts by whole periods that bring a polyline with this box over
    /// the domain.
    fn copies(&self, lo: P2, hi: P2, tolerance: f64) -> Vec<P2> {
        let range = |k: usize| -> Vec<f64> {
            match self.period[k] {
                None => vec![0.0],
                Some(t) => {
                    let first = ((self.dlo[k] - hi[k] - tolerance) / t).ceil() as i64;
                    let last = ((self.dhi[k] - lo[k] + tolerance) / t).floor() as i64;
                    (first..=last.max(first)).map(|j| j as f64 * t).collect()
                }
            }
        };
        let mut out = Vec::new();
        for du in range(0) {
            for dv in range(1) {
                out.push([du, dv]);
            }
        }
        out
    }
}

/// How fast the surface moves with each parameter, at `(u, v)`.
fn speed(s: &Surface, u: f64, v: f64) -> P2 {
    let h = 1e-6;
    let su = v3::dist(s.point(u + h, v), s.point(u - h, v)) / (2.0 * h);
    let sv = v3::dist(s.point(u, v + h), s.point(u, v - h)) / (2.0 * h);
    [su.max(1e-9), sv.max(1e-9)]
}

type Sample = ((f64, f64), V3);

/// Points along the iso-curve from `a` to `b` (parameters), close enough
/// that each chord is within `tolerance` of the surface.
fn iso_points(s: &Surface, a: (f64, f64), b: (f64, f64), tolerance: f64) -> Vec<Sample> {
    fn go(s: &Surface, a: Sample, b: Sample, tolerance: f64, depth: usize, out: &mut Vec<Sample>) {
        let m = ((a.0 .0 + b.0 .0) * 0.5, (a.0 .1 + b.0 .1) * 0.5);
        let pm = s.point(m.0, m.1);
        if depth < 12 && (depth < 3 || v3::dist(pm, lerp3(a.1, b.1, 0.5)) > tolerance) {
            go(s, a, (m, pm), tolerance, depth + 1, out);
            out.push((m, pm));
            go(s, (m, pm), b, tolerance, depth + 1, out);
        }
    }
    let (sa, sb) = ((a, s.point(a.0, a.1)), (b, s.point(b.0, b.1)));
    let mut out = vec![sa];
    // A straight iso-line needs no inner points; probe once to tell.
    let pm = s.point((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
    if v3::dist(pm, lerp3(sa.1, sb.1, 0.5)) > tolerance * 1e-3 {
        go(s, sa, sb, tolerance, 0, &mut out);
    }
    out.push(sb);
    out
}

/// One side of a wrapping face's domain along `u` at `v = at`: the rim lying
/// there, if there is one, its own vertices; otherwise the iso-curve.
fn side_along_u(
    surface: &Surface,
    rim: Option<&[(V3, (f64, f64))]>,
    at: f64,
    from: f64,
    to: f64,
    tolerance: f64,
) -> Vec<(V3, (f64, f64))> {
    let Some(rim) = rim else {
        return iso_points(surface, (from, at), (to, at), tolerance)
            .into_iter()
            .map(|(q, p)| (p, q))
            .collect();
    };
    let t = surface.period().0.unwrap_or(f64::INFINITY);
    let (lo, hi) = (from.min(to), from.max(to));
    // The rim's vertices in this period, in order of `u`.
    let mut pts: Vec<(f64, V3)> = rim
        .iter()
        .map(|(p, q)| {
            let x = if t.is_finite() {
                lo + (q.0 - lo).rem_euclid(t)
            } else {
                q.0
            };
            (x, *p)
        })
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let near = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + a.abs());
    // Where the rim is at `x`: a vertex, or on the chord between two.
    let on_rim = |x: f64| -> V3 {
        if let Some((_, p)) = pts.iter().find(|(y, _)| near(*y, x) || near(*y + t, x)) {
            return *p;
        }
        let n = pts.len();
        for i in 0..n {
            let (a, pa) = pts[i];
            let (mut b, pb) = pts[(i + 1) % n];
            if i + 1 == n {
                b += t;
            }
            let y = if x < a { x + t } else { x };
            if y >= a && y <= b && b > a {
                return lerp3(pa, pb, (y - a) / (b - a));
            }
        }
        surface.point(x, at)
    };
    let mut side = vec![(on_rim(lo), (lo, at))];
    for (x, p) in &pts {
        if *x > lo && !near(*x, lo) && *x < hi && !near(*x, hi) {
            side.push((*p, (*x, at)));
        }
    }
    side.push((on_rim(hi), (hi, at)));
    if from > to {
        side.reverse();
    }
    side
}

/// The faces of a body as patches, or `None` when one cannot be laid out.
fn patches(body: &Body, tolerance: f64) -> Option<Vec<Patch>> {
    let mut out = Vec::with_capacity(body.faces().len());
    for (fi, face) in body.faces().iter().enumerate() {
        let surface = body.surfaces()[face.surface].clone();
        let (pu, pv) = surface.period();
        let loops = body.face_loops(fi);
        let real: Vec<&TrimLoop> = loops
            .iter()
            .filter(|l| l.is_outer() || l.is_hole())
            .collect();
        let wraps = [face.u_wraps && pu.is_some(), face.v_wraps && pv.is_some()];
        let mut rings: Vec<Ring> = Vec::new();
        let anchor;
        let mut wraps = wraps;
        if !wraps[0] && !wraps[1] {
            if real.is_empty() || real.iter().any(|l| l.vertices.len() < 3) {
                return None;
            }
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for q in real.iter().flat_map(|l| l.uv.iter()) {
                for k in 0..2 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            // A face a whole period across, closed by its seam as an edge
            // it uses twice: the domain is its loops as they are, with the
            // seam on both sides.
            for (k, p) in [pu, pv].into_iter().enumerate() {
                if let Some(t) = p {
                    let span = hi[k] - lo[k];
                    if span > t * (1.0 + 1e-6) {
                        return None;
                    }
                    if span >= t * (1.0 - 1e-6) {
                        wraps[k] = true;
                    }
                }
            }
            anchor = ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5);
            let mut sorted = real.clone();
            sorted.sort_by_key(|l| !l.is_outer());
            for l in sorted {
                let pts: Vec<V3> = l.vertices.iter().map(|&v| body.vertices()[v]).collect();
                rings.push((pts, l.uv.iter().map(|q| (q[0], q[1])).collect()));
            }
        } else {
            if real.iter().any(|l| l.is_outer()) || wraps[1] && !wraps[0] {
                return None;
            }
            let rims: Vec<Vec<(V3, (f64, f64))>> = loops
                .iter()
                .filter(|l| l.is_degenerate())
                .map(|l| {
                    l.vertices
                        .iter()
                        .filter_map(|&v| {
                            let p = body.vertices()[v];
                            surface.invert(p).map(|q| (p, q))
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|r: &Vec<_>| !r.is_empty())
                .collect();
            // A period wide where the face wraps, starting at a vertex of a
            // rim so the seam's ends are vertices already there.
            let mut u = face.u_range;
            let mut v = face.v_range;
            if let Some(t) = pu {
                let start = rims.first().map(|r| r[0].1 .0).unwrap_or(u.0);
                u = (start, start + t);
            }
            if let (true, Some(t)) = (wraps[1], pv) {
                v = (v.0, v.0 + t);
            } else if !rims.is_empty() {
                // Where the rims are, not where the face says: the two can
                // part by rounding, and a side that misses its rim is sampled
                // afresh and shares no point with the face beyond it.
                let mean = |r: &Vec<(V3, (f64, f64))>| {
                    r.iter().map(|x| x.1 .1).sum::<f64>() / r.len() as f64
                };
                let heights: Vec<f64> = rims.iter().map(mean).collect();
                let (lo, hi) = heights
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |m, &h| {
                        (m.0.min(h), m.1.max(h))
                    });
                let span = (v.1 - v.0).abs().max(1e-12);
                if (lo - v.0).abs() <= span * 1e-3 {
                    v.0 = lo;
                }
                if (hi - v.1).abs() <= span * 1e-3 {
                    v.1 = hi;
                }
            }
            anchor = ((u.0 + u.1) * 0.5, (v.0 + v.1) * 0.5);
            // The rim at a height: every point of it within a thousandth of
            // the face's height of `at`.
            let reach = (v.1 - v.0).abs() * 1e-3 + 1e-9;
            let rim_at = |at: f64| {
                rims.iter()
                    .find(|r| r.iter().all(|(_, q)| (q.1 - at).abs() <= reach))
                    .map(|r| r.as_slice())
            };
            if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                eprintln!(
                    "  [fuse] wrapping {} face {fi}: u {u:?} v {v:?}, rims at {:?}, found {} {}",
                    surface.kind(),
                    rims.iter()
                        .map(|r| (r.len(), r[0].1 .1))
                        .collect::<Vec<_>>(),
                    rim_at(v.0).is_some(),
                    rim_at(v.1).is_some()
                );
            }
            let t4 = tolerance * 0.25;
            let mut ring: Vec<(V3, (f64, f64))> = Vec::new();
            for side in [
                side_along_u(&surface, rim_at(v.0), v.0, u.0, u.1, t4),
                iso_points(&surface, (u.1, v.0), (u.1, v.1), t4)
                    .into_iter()
                    .map(|(q, p)| (p, q))
                    .collect(),
                side_along_u(&surface, rim_at(v.1), v.1, u.1, u.0, t4),
                iso_points(&surface, (u.0, v.1), (u.0, v.0), t4)
                    .into_iter()
                    .map(|(q, p)| (p, q))
                    .collect(),
            ] {
                let n = side.len();
                ring.extend(side.into_iter().take(n - 1));
            }
            rings.push((
                ring.iter().map(|x| x.0).collect(),
                ring.iter().map(|x| x.1).collect(),
            ));
            for l in real {
                let pts: Vec<V3> = l.vertices.iter().map(|&v| body.vertices()[v]).collect();
                let mut uv: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
                for p in &pts {
                    let near = uv.last().copied().unwrap_or(anchor);
                    uv.push(surface.invert_near(*p, near)?);
                }
                rings.push((pts, uv));
            }
        }
        let scale = speed(&surface, anchor.0, anchor.1);
        let mut rings2: Vec<Vec<P2>> = rings
            .iter()
            .map(|(_, uv)| {
                uv.iter()
                    .map(|q| [q.0 * scale[0], q.1 * scale[1]])
                    .collect()
            })
            .collect();
        let mut rings3: Vec<Vec<V3>> = rings.into_iter().map(|(p, _)| p).collect();
        let period = [pu.map(|t| t * scale[0]), pv.map(|t| t * scale[1])];
        let (mut dlo, mut dhi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in rings2.first().into_iter().flatten() {
            for k in 0..2 {
                dlo[k] = dlo[k].min(q[k]);
                dhi[k] = dhi[k].max(q[k]);
            }
        }
        // A hole of a wrapping face that crosses its seam is also a hole a
        // period over.
        if wraps[0] || wraps[1] {
            let n = rings2.len();
            for i in 1..n {
                for k in 0..2 {
                    let Some(t) = period[k].filter(|_| wraps[k]) else {
                        continue;
                    };
                    for shift in [-t, t] {
                        let moved: Vec<P2> = rings2[i]
                            .iter()
                            .map(|q| {
                                let mut q = *q;
                                q[k] += shift;
                                q
                            })
                            .collect();
                        if moved.iter().any(|q| q[k] > dlo[k] && q[k] < dhi[k]) {
                            rings2.push(moved);
                            rings3.push(rings3[i].clone());
                        }
                    }
                }
            }
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let mut grow = |p: V3| {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        };
        for p in rings3.iter().flatten() {
            grow(*p);
        }
        // A curved face bulges past its outline — a sphere's outline is one
        // meridian — so its box is taken over the whole of its domain.
        if !matches!(surface, Surface::Plane { .. }) {
            const N: usize = 16;
            for i in 0..=N {
                for j in 0..=N {
                    let u = dlo[0] + (dhi[0] - dlo[0]) * i as f64 / N as f64;
                    let v = dlo[1] + (dhi[1] - dlo[1]) * j as f64 / N as f64;
                    grow(surface.point(u / scale[0], v / scale[1]));
                }
            }
        }
        let pad = tolerance + (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 0.02;
        let plane = match &surface {
            Surface::Plane { origin, normal, .. } => Some((*origin, *normal)),
            _ => None,
        };
        out.push(Patch {
            surface,
            flipped: face.flipped,
            plane,
            scale,
            anchor,
            period,
            rings2,
            rings3,
            dlo,
            dhi,
            lo: lo.map(|x| x - pad),
            hi: hi.map(|x| x + pad),
            wraps,
        });
    }
    Some(out)
}

fn boxes_meet(a: (V3, V3), b: (V3, V3), tolerance: f64) -> bool {
    (0..3).all(|k| a.0[k] <= b.1[k] + tolerance && b.0[k] <= a.1[k] + tolerance)
}

// -- sections ----------------------------------------------------------------

enum Section {
    Apart,
    /// The two touch without crossing: along a curve or at a point.
    Graze,
    /// The two lie on one surface; whether they face the same way.
    Wall(bool),
    Cuts(Vec<Vec<V3>>),
}

/// Where the line `p0 + t·d` (lying in the plane of `f`, `d` unit) is on the
/// face, boundary included, as intervals of `t`.
fn intervals(f: &Patch, p0: V3, d: V3, tolerance: f64) -> Vec<(f64, f64)> {
    let q0 = f.uv(p0).unwrap_or([0.0, 0.0]);
    let q1 = f.uv(v3::add(p0, d)).unwrap_or([1.0, 0.0]);
    let dd = sub2(q1, q0);
    let mut ts: Vec<f64> = Vec::new();
    for ring in &f.rings2 {
        let n = ring.len();
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (sa, sb) = (cross2(dd, sub2(a, q0)), cross2(dd, sub2(b, q0)));
            let (ta, tb) = (dot2(sub2(a, q0), dd), dot2(sub2(b, q0), dd));
            let (za, zb) = (sa.abs() <= tolerance, sb.abs() <= tolerance);
            if za {
                ts.push(ta);
            }
            if zb {
                ts.push(tb);
            }
            if !za && !zb && (sa > 0.0) != (sb > 0.0) {
                ts.push(ta + (tb - ta) * sa / (sa - sb));
            }
        }
    }
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|x, y| (*x - *y).abs() <= tolerance);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for w in ts.windows(2) {
        let m = (w[0] + w[1]) * 0.5;
        let q = [q0[0] + dd[0] * m, q0[1] + dd[1] * m];
        if ring_distance(&f.rings2, q) > tolerance && !in_rings(&f.rings2, q) {
            continue;
        }
        match out.last_mut() {
            Some(last) if (last.1 - w[0]).abs() <= tolerance => last.1 = w[1],
            _ => out.push((w[0], w[1])),
        }
    }
    out
}

/// Two planes: the line they share, clipped to both faces.
fn plane_section(f: &Patch, g: &Patch, tolerance: f64) -> Section {
    let ((fo, fnorm), (go, gnorm)) = (f.plane.unwrap(), g.plane.unwrap());
    let u = v3::cross(fnorm, gnorm);
    let lu = v3::norm(u);
    if lu <= 1e-9 {
        if v3::dot(fnorm, v3::sub(go, fo)).abs() > tolerance {
            return Section::Apart;
        }
        let same = (v3::dot(fnorm, gnorm) > 0.0) == (f.flipped == g.flipped);
        return Section::Wall(same);
    }
    let (h1, h2) = (v3::dot(fnorm, fo), v3::dot(gnorm, go));
    let p0 = v3::scale(
        v3::add(
            v3::scale(v3::cross(gnorm, u), h1),
            v3::scale(v3::cross(u, fnorm), h2),
        ),
        1.0 / (lu * lu),
    );
    let d = v3::scale(u, 1.0 / lu);
    let (a, b) = (
        intervals(f, p0, d, tolerance),
        intervals(g, p0, d, tolerance),
    );
    let mut cuts = Vec::new();
    for &(a0, a1) in &a {
        for &(b0, b1) in &b {
            let (lo, hi) = (a0.max(b0), a1.min(b1));
            if hi - lo > tolerance {
                cuts.push(vec![
                    v3::add(p0, v3::scale(d, lo)),
                    v3::add(p0, v3::scale(d, hi)),
                ]);
            }
        }
    }
    Section::Cuts(cuts)
}

/// A curve as points, within a box: `None` for a point of contact.
fn polyline(curve: &Curve3d, lo: V3, hi: V3, tolerance: f64) -> Option<(Vec<V3>, bool)> {
    use std::f64::consts::TAU;
    fn count(c: &Curve3d) -> usize {
        match c {
            Curve3d::Sampled { points, .. } => points.len(),
            Curve3d::Spline { at, .. } => at.len(),
            Curve3d::OnSurfaces { param, .. } => count(param),
            _ => 0,
        }
    }
    fn is_closed(c: &Curve3d) -> bool {
        match c {
            Curve3d::Sampled { closed, .. } | Curve3d::Spline { closed, .. } => *closed,
            Curve3d::OnSurfaces { param, .. } => is_closed(param),
            _ => false,
        }
    }
    match curve {
        Curve3d::Point(_) => None,
        Curve3d::Line { origin, dir } => {
            let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
            for k in 0..3 {
                if dir[k].abs() < 1e-15 {
                    if origin[k] < lo[k] || origin[k] > hi[k] {
                        return None;
                    }
                    continue;
                }
                let (a, b) = ((lo[k] - origin[k]) / dir[k], (hi[k] - origin[k]) / dir[k]);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
            }
            (t1 - t0 > tolerance).then(|| (vec![curve.point(t0), curve.point(t1)], false))
        }
        Curve3d::Circle { radius: r, .. } | Curve3d::Ellipse { a: r, .. } => {
            let per = 2.0
                * (1.0 - (tolerance * 0.25 / r.max(1e-12)).min(1.0))
                    .acos()
                    .max(1e-6);
            let n = ((TAU / per).ceil() as usize).clamp(12, 2048);
            Some((
                (0..n)
                    .map(|i| curve.point(TAU * i as f64 / n as f64))
                    .collect(),
                true,
            ))
        }
        _ => {
            let n = count(curve);
            (n >= 2).then(|| {
                (
                    (0..n).map(|i| curve.point(i as f64)).collect(),
                    is_closed(curve),
                )
            })
        }
    }
}

/// The stretches of a polyline lying on both patches.
fn clip(poly: &[V3], closed: bool, f: &Patch, g: &Patch, tolerance: f64) -> Vec<Vec<V3>> {
    let mut pts = poly.to_vec();
    if closed && pts.len() > 2 {
        pts.push(pts[0]);
    }
    if pts.len() < 2 {
        return Vec::new();
    }
    let on = |p: V3| f.holds(p, tolerance) && g.holds(p, tolerance);
    let extent = |p: &Patch| (0..3).map(|k| p.hi[k] - p.lo[k]).fold(0.0f64, f64::max);
    let size = extent(f).max(extent(g));
    let mut runs: Vec<Vec<V3>> = Vec::new();
    let mut cur: Vec<V3> = Vec::new();
    let first_on = on(pts[0]);
    let mut prev = (pts[0], first_on);
    if first_on {
        cur.push(pts[0]);
    }
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let m = ((v3::dist(a, b) / (size * 0.02).max(tolerance)).ceil() as usize).clamp(1, 64);
        for j in 1..=m {
            let p = lerp3(a, b, j as f64 / m as f64);
            let s = on(p);
            if s != prev.1 {
                // Where it changes, by halving: against the boundary itself
                // where the two ends disagree about it, so the end lands on
                // the boundary rather than a tolerance beyond it.
                let (mut x, mut y) = (prev.0, p);
                // Close enough to the boundary that the end welds to the point on
                // it; far enough that a run along an edge, a rounding off it,
                // still reads as on.
                let tight = tolerance * 1e-3;
                let exact = |q: V3| {
                    f.holds_within(q, tolerance, tight) && g.holds_within(q, tolerance, tight)
                };
                // Where both ends of the bracket lie within the tolerance of
                // the boundary on the same side of it — a sample can fall
                // just short — the exact boundary is a little beyond one of
                // them: widen the bracket a few tolerances and look there.
                if exact(x) == exact(y) {
                    let dir = v3::sub(y, x);
                    let l = v3::norm(dir);
                    if l > 0.0 {
                        let reach = v3::scale(dir, tolerance * 4.0 / l);
                        let (wx, wy) = (v3::sub(x, reach), v3::add(y, reach));
                        if exact(wx) != exact(y) {
                            x = wx;
                        } else if exact(x) != exact(wy) {
                            y = wy;
                        }
                    }
                }
                let test: &dyn Fn(V3) -> bool = if exact(x) != exact(y) { &exact } else { &on };
                let side = test(x);
                for _ in 0..50 {
                    let mid = lerp3(x, y, 0.5);
                    if test(mid) == side {
                        x = mid;
                    } else {
                        y = mid;
                    }
                }
                if prev.1 {
                    cur.push(x);
                    runs.push(std::mem::take(&mut cur));
                } else {
                    cur = vec![y];
                }
            }
            if j == m && s {
                cur.push(b);
            }
            prev = (p, s);
        }
    }
    let whole = runs.is_empty() && cur.len() == pts.len();
    if cur.len() >= 2 {
        runs.push(cur);
    }
    if closed && !whole && first_on && runs.len() >= 2 {
        let last = runs.pop().unwrap();
        let mut joined = last;
        joined.extend(runs[0].iter().skip(1).copied());
        runs[0] = joined;
    }
    runs.retain(|r| r.windows(2).map(|w| v3::dist(w[0], w[1])).sum::<f64>() > tolerance);
    // A stretch along a ring of either face is that ring, with its points.
    for run in &mut runs {
        if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
            let snaps: Vec<bool> = [f, g]
                .iter()
                .map(|p| {
                    p.rings3
                        .iter()
                        .any(|r| along_ring(run, r, tolerance).is_some())
                })
                .collect();
            eprintln!(
                "  [fuse] run of {} points, closes {}, along a ring of {} {:?} / {} {:?}",
                run.len(),
                v3::dist(run[0], run[run.len() - 1]) <= tolerance,
                f.surface.kind(),
                snaps[0],
                g.surface.kind(),
                snaps[1]
            );
        }
        for p in [f, g] {
            if let Some(along) = p
                .rings3
                .iter()
                .find_map(|ring| along_ring(run, ring, tolerance))
            {
                *run = along;
                break;
            }
        }
    }
    runs
}

/// When `run` lies along the closed polyline `ring`, the ring's own points
/// between the run's ends, in the run's direction.
fn along_ring(run: &[V3], ring: &[V3], tolerance: f64) -> Option<Vec<V3>> {
    let n = ring.len();
    if n < 2 || run.len() < 2 {
        return None;
    }
    let mut closed = ring.to_vec();
    closed.push(ring[0]);
    let off = |p: V3| to_polyline(p, &closed) > tolerance;
    // Every point and every middle: a chord across the ring's corner is not
    // along it.
    if run.iter().any(|p| off(*p)) || run.windows(2).any(|w| off(lerp3(w[0], w[1], 0.5))) {
        return None;
    }
    // Position along the ring, as segment index plus fraction.
    let place = |p: V3| -> f64 {
        let mut best = (f64::INFINITY, 0.0);
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let ab = v3::sub(b, a);
            let l = v3::dot(ab, ab);
            let t = if l > 0.0 {
                (v3::dot(v3::sub(p, a), ab) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d = v3::dist(p, v3::add(a, v3::scale(ab, t)));
            if d < best.0 {
                best = (d, i as f64 + t);
            }
        }
        best.1
    };
    let nf = n as f64;
    let (s, e) = (place(run[0]), place(run[run.len() - 1]));
    // Which way the run goes round: by where its middle is.
    let mid = place(lerp3(
        run[run.len() / 2 - usize::from(run.len() == 2)],
        run[run.len() / 2],
        0.5,
    ));
    let closes = v3::dist(run[0], run[run.len() - 1]) <= tolerance;
    if closes {
        // A closed run along part of a ring — a band's rim, which in the
        // band's outline runs from one side of its seam to the other — is
        // the stretch of the ring that stays on it, each way from where
        // the run starts. Going round the whole ring would take the seam
        // and the other rim with it.
        let on = |p: V3| to_polyline(p, run) <= tolerance;
        let idx = |k: f64| (k.rem_euclid(nf) as usize) % n;
        let mut ahead = Vec::new();
        let mut k = s.floor() + 1.0;
        while ahead.len() < n && on(ring[idx(k)]) {
            ahead.push(ring[idx(k)]);
            k += 1.0;
        }
        if ahead.len() >= n {
            // The ring is the run: all of it, from where the run starts.
            let mut out = vec![run[0]];
            out.extend(ahead);
            out.push(run[run.len() - 1]);
            return Some(out);
        }
        let mut behind = Vec::new();
        let mut k = s.ceil() - 1.0;
        while behind.len() + ahead.len() < n && on(ring[idx(k)]) {
            behind.push(ring[idx(k)]);
            k -= 1.0;
        }
        behind.reverse();
        let mut out = behind;
        out.push(run[0]);
        out.extend(ahead);
        out.dedup_by(|a, b| v3::dist(*a, *b) <= tolerance * 1e-6);
        return (out.len() >= 2).then_some(out);
    }
    let forward = if closes {
        (mid - s).rem_euclid(nf) < nf * 0.5
    } else {
        (mid - s).rem_euclid(nf) <= (e - s).rem_euclid(nf)
    };
    let span = if closes {
        nf
    } else if forward {
        (e - s).rem_euclid(nf)
    } else {
        (s - e).rem_euclid(nf)
    };
    let mut out = vec![run[0]];
    let mut k = if forward {
        s.floor() + 1.0
    } else {
        s.ceil() - 1.0
    };
    loop {
        let d = if forward { k - s } else { s - k };
        if d >= span - 1e-9 {
            break;
        }
        if d > 1e-9 {
            out.push(ring[(k.rem_euclid(nf) as usize) % n]);
        }
        k += if forward { 1.0 } else { -1.0 };
    }
    out.push(run[run.len() - 1]);
    // The ring's points taken must be on the run: a closed run along one
    // side of a ring is not the whole ring.
    if out.iter().any(|p| to_polyline(*p, run) > tolerance) {
        return None;
    }
    Some(out)
}

enum Touch {
    Crosses,
    AllAlong,
    Somewhere,
}

/// Do the two surfaces cross along this run, or touch?
///
/// The sine between their normals is near zero where they touch, and the
/// way they lean across the run changes sign at a touch lying between two
/// samples, which no threshold on either sample sees: at twice the size, the
/// same graze of two equal cylinders fell between samples and was missed.
fn touch(run: &[V3], f: &Patch, g: &Patch) -> Touch {
    let normal = |p: &Patch, x: V3| {
        p.uv(x)
            .and_then(|q| p.surface.normal(q[0] / p.scale[0], q[1] / p.scale[1]))
    };
    let mut flat_all_along = run.len() >= 2;
    let mut before: Option<(f64, f64)> = None;
    for w in run.windows(2) {
        let (Some(a), Some(b)) = (normal(f, w[0]), normal(g, w[0])) else {
            before = None;
            flat_all_along = false;
            continue;
        };
        let x = v3::cross(a, b);
        let sine = v3::norm(x);
        let lean = v3::dot(x, v3::sub(w[1], w[0]));
        if sine >= 1e-6 {
            flat_all_along = false;
        }
        if sine < 1e-2 {
            before = Some((sine, lean));
            continue;
        }
        if let Some((was, dir)) = before
            && dir * lean < 0.0
            && was.min(sine) < 0.2
        {
            return Touch::Somewhere;
        }
        before = Some((sine, lean));
    }
    if flat_all_along {
        return Touch::AllAlong;
    }
    let touches = run.iter().any(|x| {
        matches!((normal(f, *x), normal(g, *x)), (Some(a), Some(b)) if v3::norm(v3::cross(a, b)) < 1e-2)
    });
    if touches {
        Touch::Somewhere
    } else {
        Touch::Crosses
    }
}

/// One face's ring handed to another on the same surface, as stretches,
/// each stretch that runs along a ring of the other taking that ring's own
/// points. The two solids sample one circle each their own way, and laid
/// side by side the two polygons leave slivers between them.
fn onto_rings(ring: &[V3], to: &Patch, tolerance: f64) -> Vec<Vec<V3>> {
    let n = ring.len();
    let near = |p: V3| {
        to.rings3.iter().any(|r| {
            let mut c = r.clone();
            c.push(r[0]);
            to_polyline(p, &c) <= tolerance
        })
    };
    // Whether each side lies along the other's rings.
    let along: Vec<bool> = (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            near(a) && near(b) && near(lerp3(a, b, 0.5))
        })
        .collect();
    let mut closed = ring.to_vec();
    closed.push(ring[0]);
    if along.iter().all(|&x| !x) {
        return vec![closed];
    }
    if along.iter().all(|&x| x) {
        // Wholly along one of the other's rings: nothing new to lay down.
        return Vec::new();
    }
    // Stretches between changes, starting at one.
    let start = (0..n)
        .find(|&i| along[i] != along[(i + n - 1) % n])
        .unwrap_or(0);
    let mut out = Vec::new();
    let mut i = start;
    let mut run = vec![ring[i]];
    let mut kind = along[i];
    for _ in 0..n {
        let next = (i + 1) % n;
        run.push(ring[next]);
        i = next;
        if along[i] != kind || i == start {
            if !kind {
                out.push(std::mem::take(&mut run));
            } else {
                // Along the other's ring: its own points, between the ends.
                let snapped = to
                    .rings3
                    .iter()
                    .find_map(|r| along_ring(&run, r, tolerance));
                out.push(snapped.unwrap_or_else(|| run.clone()));
                run.clear();
            }
            run = vec![ring[i]];
            kind = along[i];
        }
    }
    out
}

fn section(f: &Patch, g: &Patch, tolerance: f64) -> Section {
    if !boxes_meet((f.lo, f.hi), (g.lo, g.hi), tolerance) {
        return Section::Apart;
    }
    if f.plane.is_some() && g.plane.is_some() {
        return plane_section(f, g, tolerance);
    }
    let lo = [0, 1, 2].map(|k| f.lo[k].max(g.lo[k]));
    let hi = [0, 1, 2].map(|k| f.hi[k].min(g.hi[k]));
    let curves = match ssi(&f.surface, &g.surface) {
        SsiResult::Disjoint => return Section::Apart,
        SsiResult::Coincident { opposite } => {
            return Section::Wall(!(opposite ^ (f.flipped != g.flipped)));
        }
        SsiResult::Curves(cs) => cs,
        SsiResult::Unknown => march(&f.surface, &g.surface, (lo, hi), tolerance * 0.25),
    };
    if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
        for c in &curves {
            if let Some((pts, closed)) = polyline(c, lo, hi, tolerance) {
                eprintln!(
                    "  [fuse] {} x {}: {} curve of {} points, closed {closed}, {:?} .. {:?}",
                    f.surface.kind(),
                    g.surface.kind(),
                    c.kind(),
                    pts.len(),
                    pts.first(),
                    pts.last()
                );
            }
        }
    }
    let mut cuts = Vec::new();
    for c in &curves {
        if let Curve3d::Point(p) = c {
            if f.holds(*p, tolerance) && g.holds(*p, tolerance) {
                return Section::Graze;
            }
            continue;
        }
        if let Some((pts, closed)) = polyline(c, lo, hi, tolerance) {
            if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                let p = pts[0];
                let gap = |x: &Patch| x.uv(p).map(|q| ring_distance(&x.rings2, q));
                eprintln!(
                    "  [fuse] its first point: on {} {} ({:?} from its boundary), on {} {} ({:?})",
                    f.surface.kind(),
                    f.holds(p, tolerance),
                    gap(f),
                    g.surface.kind(),
                    g.holds(p, tolerance),
                    gap(g)
                );
            }
            let runs = clip(&pts, closed, f, g, tolerance);
            if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                eprintln!(
                    "  [fuse] clipped to {} runs of {:?} points",
                    runs.len(),
                    runs.iter().map(|r| r.len()).collect::<Vec<_>>()
                );
            }
            for run in runs {
                match touch(&run, f, g) {
                    Touch::Crosses => cuts.push(run),
                    // Resting along a ruling: neither face is cut.
                    Touch::AllAlong => {}
                    Touch::Somewhere => return Section::Graze,
                }
            }
        }
    }
    Section::Cuts(cuts)
}

// -- the arrangement of one face ---------------------------------------------

/// Points welded at a tolerance: the same place is the same index.
struct Pool {
    points: Vec<V3>,
    grid: HashMap<[i64; 3], Vec<usize>>,
    tolerance: f64,
}

impl Pool {
    fn new(tolerance: f64) -> Self {
        Self {
            points: Vec::new(),
            grid: HashMap::new(),
            tolerance,
        }
    }

    fn id(&mut self, p: V3) -> usize {
        let k = p.map(|x| (x / self.tolerance).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = self.grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for &i in list {
                            if v3::dist(self.points[i], p) <= self.tolerance {
                                return i;
                            }
                        }
                    }
                }
            }
        }
        self.points.push(p);
        let i = self.points.len() - 1;
        self.grid.entry(k).or_default().push(i);
        i
    }
}

/// A vertex of a face's arrangement: which point in space, and where in the
/// face's parameters. One point can be two vertices — the two ends of a seam,
/// the corners at a pole.
#[derive(Clone, Copy)]
struct Node {
    point: usize,
    uv: P2,
}

/// One piece of a face: its outline and holes as nodes, and a point inside it.
struct Cell {
    rings: Vec<Vec<Node>>,
    probe: V3,
}

/// The point of a region furthest from its boundary, from a grid over it.
fn probe_of(rings: &[Vec<P2>], tolerance: f64) -> Option<P2> {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for q in &rings[0] {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let mut best: Option<(f64, P2)> = None;
    for n in [7usize, 23, 61] {
        for i in 0..n {
            for j in 0..n {
                let q = [
                    lo[0] + (hi[0] - lo[0]) * (i as f64 + 0.5) / n as f64,
                    lo[1] + (hi[1] - lo[1]) * (j as f64 + 0.5) / n as f64,
                ];
                if !in_rings(rings, q) {
                    continue;
                }
                let d = ring_distance(rings, q);
                if best.is_none_or(|b| d > b.0) {
                    best = Some((d, q));
                }
            }
        }
        if best.is_some_and(|b| b.0 > tolerance * 4.0) {
            break;
        }
    }
    best.filter(|b| b.0 > tolerance * 0.1).map(|b| b.1)
}

/// Nodes of one face's arrangement, one per point and place in parameters.
struct Nodes {
    list: Vec<Node>,
    by_point: HashMap<usize, Vec<usize>>,
    tolerance: f64,
}

impl Nodes {
    fn id(&mut self, point: usize, uv: P2) -> usize {
        let list = self.by_point.entry(point).or_default();
        for &n in list.iter() {
            if len2(sub2(self.list[n].uv, uv)) <= self.tolerance * 4.0 {
                return n;
            }
        }
        self.list.push(Node { point, uv });
        list.push(self.list.len() - 1);
        self.list.len() - 1
    }
}

/// Boxes in the plane, filed by the cells of a uniform grid they cover.
struct Grid2 {
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
    /// Boxes too big to file, offered to every query.
    big: Vec<usize>,
}

impl Grid2 {
    fn of(boxes: &[(P2, P2)], tolerance: f64) -> Grid2 {
        // Cells about the size of a typical box.
        let mut sizes: Vec<f64> = boxes
            .iter()
            .map(|b| (b.1[0] - b.0[0]).max(b.1[1] - b.0[1]))
            .collect();
        sizes.sort_by(f64::total_cmp);
        let cell = sizes
            .get(sizes.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(tolerance * 4.0)
            .max(1e-12);
        let mut grid = Grid2 {
            cell,
            cells: HashMap::new(),
            big: Vec::new(),
        };
        for (i, b) in boxes.iter().enumerate() {
            let (lo, hi) = grid.span(*b, tolerance);
            if (hi.0 - lo.0 + 1) * (hi.1 - lo.1 + 1) > 256 {
                grid.big.push(i);
                continue;
            }
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    grid.cells.entry((x, y)).or_default().push(i);
                }
            }
        }
        grid
    }

    fn span(&self, b: (P2, P2), pad: f64) -> ((i64, i64), (i64, i64)) {
        let k = |x: f64| (x / self.cell).floor() as i64;
        (
            (k(b.0[0] - pad), k(b.0[1] - pad)),
            (k(b.1[0] + pad), k(b.1[1] + pad)),
        )
    }

    /// Every box filed near `b`, some more than once.
    fn near(&self, b: (P2, P2), pad: f64) -> Vec<usize> {
        let (lo, hi) = self.span(b, pad);
        let mut out = self.big.clone();
        if (hi.0 - lo.0 + 1) * (hi.1 - lo.1 + 1) > 4096 {
            // A box as big as the face: everything.
            let mut all: Vec<usize> = self.cells.values().flatten().copied().collect();
            all.extend(&self.big);
            all.sort_unstable();
            all.dedup();
            return all;
        }
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                if let Some(list) = self.cells.get(&(x, y)) {
                    out.extend(list);
                }
            }
        }
        out
    }
}

/// Cut a patch by its sections and return its pieces.
fn cells(
    f: &Patch,
    index: usize,
    cuts: &[Vec<V3>],
    pool: &mut Pool,
    tolerance: f64,
) -> Result<Vec<Cell>, Declined> {
    let mut nodes = Nodes {
        list: Vec::new(),
        by_point: HashMap::new(),
        tolerance,
    };
    let mut segs: Vec<(usize, usize)> = Vec::new();
    for (pts, uv) in f.rings3.iter().zip(&f.rings2) {
        let ids: Vec<usize> = pts
            .iter()
            .zip(uv)
            .map(|(p, q)| nodes.id(pool.id(*p), *q))
            .collect();
        let n = ids.len();
        for i in 0..n {
            segs.push((ids[i], ids[(i + 1) % n]));
        }
    }
    let pad = tolerance * 2.0;
    for cut in cuts {
        let Some(uv) = f.unwrapped(cut) else { continue };
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in &uv {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        let ids: Vec<usize> = cut.iter().map(|p| pool.id(*p)).collect();
        for shift in f.copies(lo, hi, tolerance) {
            let moved: Vec<P2> = uv
                .iter()
                .map(|q| [q[0] + shift[0], q[1] + shift[1]])
                .collect();
            let ns: Vec<usize> = ids
                .iter()
                .zip(&moved)
                .map(|(&p, q)| nodes.id(p, *q))
                .collect();
            for (k, w) in ns.windows(2).enumerate() {
                let (a, b) = (moved[k], moved[k + 1]);
                let outside = (0..2).any(|d| {
                    (a[d] < f.dlo[d] - pad && b[d] < f.dlo[d] - pad)
                        || (a[d] > f.dhi[d] + pad && b[d] > f.dhi[d] + pad)
                });
                if !outside {
                    segs.push((w[0], w[1]));
                }
            }
        }
    }
    segs.retain(|(a, b)| a != b);

    // Where segments cross, each crossing a node of both.
    let bbox = |s: (usize, usize), nodes: &Nodes| {
        let (a, b) = (nodes.list[s.0].uv, nodes.list[s.1].uv);
        (
            [a[0].min(b[0]), a[1].min(b[1])],
            [a[0].max(b[0]), a[1].max(b[1])],
        )
    };
    let boxes: Vec<(P2, P2)> = segs.iter().map(|s| bbox(*s, &nodes)).collect();
    let mut splits: Vec<Vec<usize>> = vec![Vec::new(); segs.len()];
    // Pairs whose boxes meet, from a grid over them rather than every pair:
    // a fine tolerance samples a face's curves into thousands of segments.
    let grid = Grid2::of(&boxes, tolerance);
    let mut seen = vec![usize::MAX; segs.len()];
    for i in 0..segs.len() {
        for j in grid.near(boxes[i], tolerance) {
            if j <= i || seen[j] == i {
                continue;
            }
            seen[j] = i;
            let (bi, bj) = (boxes[i], boxes[j]);
            if bi.0[0] > bj.1[0] + tolerance
                || bj.0[0] > bi.1[0] + tolerance
                || bi.0[1] > bj.1[1] + tolerance
                || bj.0[1] > bi.1[1] + tolerance
            {
                continue;
            }
            let (a, b) = (nodes.list[segs[i].0].uv, nodes.list[segs[i].1].uv);
            let (c, d) = (nodes.list[segs[j].0].uv, nodes.list[segs[j].1].uv);
            let (r, s) = (sub2(b, a), sub2(d, c));
            let (lr, ls) = (len2(r), len2(s));
            let denom = cross2(r, s);
            if denom.abs() <= 1e-12 * lr * ls {
                continue;
            }
            let ac = sub2(c, a);
            let t = cross2(ac, s) / denom;
            let u = cross2(ac, r) / denom;
            let (et, eu) = (tolerance / lr, tolerance / ls);
            if t > et && t < 1.0 - et && u > eu && u < 1.0 - eu {
                let q = [a[0] + r[0] * t, a[1] + r[1] * t];
                let p = if f.plane.is_some() {
                    let (pa, pb) = (nodes.list[segs[i].0].point, nodes.list[segs[i].1].point);
                    lerp3(pool.points[pa], pool.points[pb], t)
                } else {
                    f.point(q)
                };
                let id = nodes.id(pool.id(p), q);
                splits[i].push(id);
                splits[j].push(id);
            }
        }
    }
    // And wherever a node lies on a segment: a touch, an overlap, an end.
    let points: Vec<(P2, P2)> = nodes.list.iter().map(|n| (n.uv, n.uv)).collect();
    let near_nodes = Grid2::of(&points, tolerance);
    for (i, &(a, b)) in segs.iter().enumerate() {
        let (pa, pb) = (nodes.list[a].uv, nodes.list[b].uv);
        let l = len2(sub2(pb, pa));
        let bi = boxes[i];
        for v in near_nodes.near(bi, tolerance) {
            if v == a || v == b {
                continue;
            }
            let q = nodes.list[v].uv;
            if q[0] < bi.0[0] - tolerance
                || q[0] > bi.1[0] + tolerance
                || q[1] < bi.0[1] - tolerance
                || q[1] > bi.1[1] + tolerance
            {
                continue;
            }
            let (d, t) = to_segment(q, pa, pb);
            if d <= tolerance && t * l > tolerance && (1.0 - t) * l > tolerance {
                splits[i].push(v);
            }
        }
    }
    let nodes = nodes.list;
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for (i, &(a, b)) in segs.iter().enumerate() {
        let pa = nodes[a].uv;
        let dir = sub2(nodes[b].uv, pa);
        let mut along = splits[i].clone();
        along.sort_by(|&x, &y| {
            dot2(sub2(nodes[x].uv, pa), dir).total_cmp(&dot2(sub2(nodes[y].uv, pa), dir))
        });
        along.dedup();
        let chain: Vec<usize> = std::iter::once(a)
            .chain(along)
            .chain(std::iter::once(b))
            .collect();
        for w in chain.windows(2) {
            if w[0] != w[1] {
                edges.insert((w[0].min(w[1]), w[0].max(w[1])));
            }
        }
    }
    // A cut that ends off the face — a wall's outline reaching past it —
    // bounds nothing there. One that ends *inside* the face is a section
    // that stopped short, a trace that could not get through a pinch, and
    // the pieces either side of it are not pieces at all.
    loop {
        let mut degree: HashMap<usize, usize> = HashMap::new();
        for &(a, b) in &edges {
            *degree.entry(a).or_default() += 1;
            *degree.entry(b).or_default() += 1;
        }
        for (&v, &d) in &degree {
            let q = nodes[v].uv;
            if d == 1 && in_rings(&f.rings2, q) && ring_distance(&f.rings2, q) > tolerance * 4.0 {
                if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                    let nearest = nodes
                        .iter()
                        .enumerate()
                        .filter(|(w, _)| *w != v)
                        .map(|(_, n)| len2(sub2(n.uv, q)))
                        .fold(f64::INFINITY, f64::min);
                    eprintln!(
                        "  [fuse] {} face {index}: a cut ends inside at {:?} ({:.3e} from its boundary, {nearest:.3e} from the nearest other point, tolerance {tolerance:.3e})",
                        f.surface.kind(),
                        pool.points[nodes[v].point],
                        ring_distance(&f.rings2, q)
                    );
                }
                return Err(Declined::NeedsArrangement { face: index });
            }
        }
        let before = edges.len();
        edges.retain(|(a, b)| degree[a] > 1 && degree[b] > 1);
        if edges.len() == before {
            break;
        }
    }

    // The graph's faces, by the tightest turn at each vertex.
    let mut edges: Vec<(usize, usize)> = edges.into_iter().collect();
    edges.sort_unstable();
    let mut half: Vec<(usize, usize)> = Vec::with_capacity(edges.len() * 2);
    for &(a, b) in &edges {
        half.push((a, b));
        half.push((b, a));
    }
    let angle = |h: usize| {
        let d = sub2(nodes[half[h].1].uv, nodes[half[h].0].uv);
        d[1].atan2(d[0])
    };
    let mut out_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (h, &(from, _)) in half.iter().enumerate() {
        out_of.entry(from).or_default().push(h);
    }
    for list in out_of.values_mut() {
        list.sort_by(|&x, &y| angle(x).total_cmp(&angle(y)));
    }
    let mut seen = vec![false; half.len()];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..half.len() {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut h = start;
        while !seen[h] {
            seen[h] = true;
            cycle.push(half[h].0);
            let back = h ^ 1;
            let list = &out_of[&half[back].0];
            let pos = list.iter().position(|&x| x == back).unwrap_or(0);
            h = list[(pos + list.len() - 1) % list.len()];
        }
        if cycle.len() >= 3 {
            cycles.push(cycle);
        }
    }

    let tiny = tolerance * tolerance;
    let mut faces: Vec<(Vec<usize>, Vec<P2>, f64)> = Vec::new();
    let mut outlines: Vec<Vec<usize>> = Vec::new();
    for c in cycles {
        let ring: Vec<P2> = c.iter().map(|&i| nodes[i].uv).collect();
        let area = signed_area(&ring);
        if area > tiny {
            faces.push((c, ring, area));
        } else if area < -tiny {
            outlines.push(c);
        }
    }
    // An outline is a hole of the smallest piece around it.
    let mut holes: Vec<Vec<Vec<usize>>> = vec![Vec::new(); faces.len()];
    for o in outlines {
        let p = nodes[o[0]].uv;
        let around = faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                let r = std::slice::from_ref(&face.1);
                in_rings(r, p) && ring_distance(r, p) > tolerance
            })
            .min_by(|x, y| x.1 .2.total_cmp(&y.1 .2))
            .map(|(i, _)| i);
        if let Some(i) = around {
            holes[i].push(o);
        }
    }
    let mut out = Vec::new();
    for ((outer, ring, _), holes) in faces.into_iter().zip(holes) {
        let mut rings2 = vec![ring];
        rings2.extend(
            holes
                .iter()
                .map(|h| h.iter().map(|&i| nodes[i].uv).collect::<Vec<P2>>()),
        );
        let Some(q) = probe_of(&rings2, tolerance) else {
            continue;
        };
        if !in_rings(&f.rings2, q) || ring_distance(&f.rings2, q) <= tolerance * 0.5 {
            continue;
        }
        let rings = std::iter::once(&outer)
            .chain(&holes)
            .map(|r| r.iter().map(|&i| nodes[i]).collect())
            .collect();
        out.push(Cell {
            rings,
            probe: f.point(q),
        });
    }
    Ok(out)
}

// -- classification ----------------------------------------------------------

/// Is `p` inside the solid bounded by these flat patches? By the parity of a
/// ray, cast again in another direction when it grazes an edge.
fn inside_flat(p: V3, faces: &[Patch], tolerance: f64) -> Option<bool> {
    // Directions no model is likely to line an edge up with.
    const DIRECTIONS: [V3; 5] = [
        [0.531_9, 0.627_3, 0.569_1],
        [-0.419_7, 0.733_1, 0.534_2],
        [0.313_7, -0.681_9, 0.660_3],
        [-0.645_1, -0.359_3, -0.674_3],
        [0.152_9, 0.297_1, -0.942_5],
    ];
    'direction: for d in DIRECTIONS {
        let d = v3::normalize(d)?;
        let mut crossings = 0;
        for f in faces {
            let (o, n) = f.plane?;
            let dn = v3::dot(n, d);
            let dist = v3::dot(n, v3::sub(p, o));
            if dn.abs() <= 1e-9 {
                if dist.abs() <= tolerance && f.holds(p, tolerance) {
                    continue 'direction;
                }
                continue;
            }
            let t = -dist / dn;
            if t <= 0.0 {
                continue;
            }
            let Some(q) = f.uv(v3::add(p, v3::scale(d, t))) else {
                continue 'direction;
            };
            if ring_distance(&f.rings2, q) <= tolerance {
                continue 'direction;
            }
            if in_rings(&f.rings2, q) {
                crossings += 1;
            }
        }
        return Some(crossings % 2 == 1);
    }
    None
}

/// Where the ray `p + t·d` (`d` unit) meets a surface, as values of `t`;
/// `None` for a surface without a closed form for it.
fn ray_hits(surface: &Surface, p: V3, d: V3) -> Option<Vec<f64>> {
    // Roots of a·t² + b·t + c.
    let roots = |a: f64, b: f64, c: f64| -> Vec<f64> {
        if a.abs() <= 1e-15 {
            return if b.abs() <= 1e-15 {
                Vec::new()
            } else {
                vec![-c / b]
            };
        }
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return Vec::new();
        }
        let r = disc.sqrt();
        // The form that loses no digits to cancellation.
        let q = -0.5 * (b + b.signum() * r);
        let mut out = vec![q / a];
        if q != 0.0 {
            out.push(c / q);
        }
        out
    };
    // The part of `v` square to `axis`.
    let flat = |v: V3, axis: V3| v3::sub(v, v3::scale(axis, v3::dot(v, axis)));
    Some(match surface {
        Surface::Plane { origin, normal, .. } => {
            let dn = v3::dot(*normal, d);
            if dn.abs() <= 1e-15 {
                Vec::new()
            } else {
                vec![v3::dot(*normal, v3::sub(*origin, p)) / dn]
            }
        }
        Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } => {
            let (w, dp) = (flat(v3::sub(p, *origin), *axis), flat(d, *axis));
            roots(
                v3::dot(dp, dp),
                2.0 * v3::dot(w, dp),
                v3::dot(w, w) - radius * radius,
            )
        }
        Surface::Sphere { center, radius, .. } => {
            let w = v3::sub(p, *center);
            roots(1.0, 2.0 * v3::dot(w, d), v3::dot(w, w) - radius * radius)
        }
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } => {
            // |radial|² = (h·tan α)², both nappes; the face's region keeps
            // the one it is on.
            let w = v3::sub(p, *apex);
            let (h, dh) = (v3::dot(w, *axis), v3::dot(d, *axis));
            let (wr, dr) = (flat(w, *axis), flat(d, *axis));
            let k = half_angle.tan().powi(2);
            roots(
                v3::dot(dr, dr) - k * dh * dh,
                2.0 * (v3::dot(wr, dr) - k * h * dh),
                v3::dot(wr, wr) - k * h * h,
            )
        }
        _ => return None,
    })
}

/// Is `p` inside the solid bounded by these patches? By the parity of a ray
/// against the surfaces themselves, cast again in another direction when it
/// grazes a face or passes by the edge of one. `None` where a surface has no
/// closed form, or no direction is clean.
fn inside_exact(p: V3, faces: &[Patch], tolerance: f64) -> Option<bool> {
    const DIRECTIONS: [V3; 7] = [
        [0.531_9, 0.627_3, 0.569_1],
        [-0.419_7, 0.733_1, 0.534_2],
        [0.313_7, -0.681_9, 0.660_3],
        [-0.645_1, -0.359_3, -0.674_3],
        [0.152_9, 0.297_1, -0.942_5],
        [0.871_3, -0.211_7, -0.442_6],
        [-0.236_1, -0.918_4, 0.317_9],
    ];
    'direction: for d in DIRECTIONS {
        let d = v3::normalize(d)?;
        let mut crossings = 0;
        for f in faces {
            for t in ray_hits(&f.surface, p, d)? {
                if t <= -tolerance {
                    continue;
                }
                let h = v3::add(p, v3::scale(d, t));
                let q = f.uv(h)?;
                let images = f.images(q);
                let near = images
                    .iter()
                    .any(|s| ring_distance(&f.rings2, *s) <= tolerance);
                let inside = images.iter().any(|s| in_rings(&f.rings2, *s));
                if !inside && !near {
                    continue;
                }
                // On the surface where the ray starts, by an edge of a face,
                // or along a face: no count is to be trusted.
                if t <= tolerance || near {
                    continue 'direction;
                }
                let normal = f.surface.normal(q[0] / f.scale[0], q[1] / f.scale[1]);
                if normal.is_none_or(|n| v3::dot(n, d).abs() <= 1e-6) {
                    continue 'direction;
                }
                crossings += 1;
            }
        }
        return Some(crossings % 2 == 1);
    }
    None
}

enum Solid<'a> {
    Flat(&'a [Patch]),
    /// Bounded by surfaces a ray meets in closed form.
    Exact(&'a [Patch]),
    Mesh(Vec<[V3; 3]>),
}

impl Solid<'_> {
    fn contains(&self, p: V3, tolerance: f64) -> Option<bool> {
        match self {
            Solid::Flat(faces) => inside_flat(p, faces, tolerance),
            Solid::Exact(faces) => inside_exact(p, faces, tolerance),
            Solid::Mesh(tris) => Some(point_in_mesh(tris, p)),
        }
    }
}

// -- the operation -----------------------------------------------------------

fn is_flat(x: &Body) -> bool {
    x.faces()
        .iter()
        .all(|f| x.surfaces()[f.surface].kind() == "plane")
}

/// A boolean through the general fuse, or `None` when a face of either solid
/// cannot be laid out as a region of its surface.
pub fn general(
    a: &Body,
    b: &Body,
    op: BooleanOp,
    tolerance: f64,
) -> Option<Result<Body, Declined>> {
    // Between flat faces nothing is sampled, so what the tolerance is for —
    // a chord's sag — does not arise, and welding at it only merges what is
    // thinner than it: a wall's 4 mm membrane cut at a tolerance of 4.7 came
    // back a solid of the wrong volume. Points there are as exact as the
    // arithmetic, and as the inputs' own vertices agree.
    let size = {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in a.vertices().iter().chain(b.vertices()) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (0..3)
            .map(|k| hi[k] - lo[k])
            .fold(0.0f64, f64::max)
            .max(tolerance)
    };
    let tolerance = if is_flat(a) && is_flat(b) {
        // A millionth of the model: fine enough to keep any feature, coarse
        // enough for inputs whose vertices agree only as far as single
        // precision carried them — a face at y = 2199.99949 against one at
        // 2200, five ten-thousandths apart on a model five metres across.
        tolerance.min((size * 1e-6).max(1e-9))
    } else {
        // Curved faces are sampled, and the tolerance is what sampling is
        // held to and what points are welded at alike: anything thinner is
        // merged. A caller wanting a building's millimetres asks for them.
        tolerance
    };
    // Curved faces are laid out from their edges, so the edges first need to
    // be as close to the curves as the tolerance asks.
    let refined = |x: &Body| {
        let mut x = x.clone();
        if !is_flat(&x) {
            x.refine_edges(tolerance * 0.25);
        }
        x
    };
    let (ra, rb) = (refined(a), refined(b));
    let pa = patches(&ra, tolerance)?;
    let pb = patches(&rb, tolerance)?;
    let origins = Origins::of(&[&ra, &rb], tolerance);
    fn solid<'a>(x: &Body, p: &'a [Patch], tolerance: f64, size: f64) -> Option<Solid<'a>> {
        if is_flat(x) {
            return Some(Solid::Flat(p));
        }
        if p.iter().all(|f| {
            matches!(
                f.surface,
                Surface::Plane { .. }
                    | Surface::Cylinder { .. }
                    | Surface::Sphere { .. }
                    | Surface::Cone { .. }
            )
        }) {
            return Some(Solid::Exact(p));
        }
        // A closed mesh to cast rays at, however coarse it takes: one that
        // does not close has no inside, and every point reads as outside it.
        // A body cut several times over can fill its faces at a few
        // ten-thousandths of its size and not at a millionth.
        [tolerance, tolerance * 10.0, tolerance * 100.0, size * 1e-3]
            .into_iter()
            .map(|t| x.solid_triangles(t))
            .find(|tris| !tris.is_empty())
            .map(Solid::Mesh)
    }
    let Some(sa) = solid(&ra, &pa, tolerance, size) else {
        return Some(Err(Declined::NotWatertight { open_edges: 0 }));
    };
    let Some(sb) = solid(&rb, &pb, tolerance, size) else {
        return Some(Err(Declined::NotWatertight { open_edges: 0 }));
    };
    Some(fuse(&pa, &pb, &sa, &sb, &origins, op, tolerance))
}

/// A boolean of two solids bounded by planes alone, or `None` when either
/// has a face that is not flat.
pub fn planar(a: &Body, b: &Body, op: BooleanOp, tolerance: f64) -> Option<Result<Body, Declined>> {
    if !is_flat(a) || !is_flat(b) {
        return None;
    }
    general(a, b, op, tolerance)
}

fn fuse(
    pa: &[Patch],
    pb: &[Patch],
    sa: &Solid,
    sb: &Solid,
    origins: &Origins,
    op: BooleanOp,
    tolerance: f64,
) -> Result<Body, Declined> {
    let na = pa.len();
    let all: Vec<&Patch> = pa.iter().chain(pb).collect();
    let mut cuts: Vec<Vec<Vec<V3>>> = vec![Vec::new(); all.len()];
    // Patches of the other solid each one shares its surface with, and
    // whether the two face the same way.
    let mut walls: Vec<Vec<(usize, bool)>> = vec![Vec::new(); all.len()];

    for i in 0..na {
        for j in na..all.len() {
            match section(all[i], all[j], tolerance) {
                Section::Apart => {}
                Section::Graze => {
                    return Err(Declined::TangentialContact {
                        face_a: i,
                        face_b: j - na,
                    });
                }
                Section::Wall(same) => {
                    if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                        eprintln!(
                            "  [fuse] faces {i} and {j} share a {} (same way {same})",
                            all[i].surface.kind()
                        );
                    }
                    walls[i].push((j, same));
                    walls[j].push((i, same));
                    for (from, to) in [(j, i), (i, j)] {
                        for ring in &all[from].rings3 {
                            cuts[to].extend(onto_rings(ring, all[to], tolerance));
                        }
                    }
                }
                Section::Cuts(cs) => {
                    for c in cs {
                        cuts[i].push(c.clone());
                        cuts[j].push(c);
                    }
                }
            }
        }
    }

    let mut pool = Pool::new(tolerance);
    // Input vertices first, so a point both solids have is theirs.
    for f in &all {
        for p in f.rings3.iter().flatten() {
            pool.id(*p);
        }
    }
    let mut kept: Vec<Kept> = Vec::new();
    for (i, f) in all.iter().enumerate() {
        let from_a = i < na;
        let other = if from_a { sb } else { sa };
        let pieces = cells(f, i, &cuts[i], &mut pool, tolerance)?;
        if std::env::var_os("THREERS_BREP_DEBUG").is_some() && !f.plane.is_some() {
            eprintln!(
                "  [fuse] {} face {i} of {}: {} pieces from {} cuts",
                f.surface.kind(),
                if from_a { "A" } else { "B" },
                pieces.len(),
                cuts[i].len()
            );
        }
        for cell in pieces {
            let on_wall = walls[i]
                .iter()
                .find_map(|&(j, same)| all[j].interior(cell.probe, tolerance).then_some(same));
            let (keep, flip) = match on_wall {
                Some(same) => match (op, from_a, same) {
                    (BooleanOp::Union, true, true) | (BooleanOp::Intersection, true, true) => {
                        (true, false)
                    }
                    (BooleanOp::Difference, true, false) => (true, false),
                    _ => (false, false),
                },
                None => {
                    let within = other
                        .contains(cell.probe, tolerance)
                        .ok_or(Declined::UnclassifiablePiece { face: i })?;
                    match (op, from_a) {
                        (BooleanOp::Union, _) => (!within, false),
                        (BooleanOp::Intersection, _) => (within, false),
                        (BooleanOp::Difference, true) => (!within, false),
                        (BooleanOp::Difference, false) => (within, true),
                    }
                }
            };
            if std::env::var_os("THREERS_BREP_DEBUG").is_some() && !from_a && f.plane.is_none() {
                eprintln!(
                    "  [fuse] piece of {} face {i}: probe {:?} on a wall {:?} keep {keep}",
                    f.surface.kind(),
                    cell.probe,
                    on_wall
                );
            }
            if keep {
                kept.push(Kept {
                    patch: i,
                    flip,
                    rings: cell.rings,
                });
            }
        }
    }
    let kept = rejoin_seams(&all, kept, tolerance);
    assemble(&all, kept, &pool, origins, tolerance)
}

struct Kept {
    patch: usize,
    flip: bool,
    rings: Vec<Vec<Node>>,
}

/// A side of a piece: piece, ring, index in the ring.
type SideAt = (usize, usize, usize);

/// Pieces of a wrapping face that meet across its seam are one face again.
///
/// Each piece is moved by whole periods so that it continues its neighbour
/// across the seam, and the seam sides the two then share are dropped. A
/// piece that reaches itself that way goes all the way round: it keeps the
/// seam, which it then uses twice, as OpenCASCADE has a cylinder's wall.
fn rejoin_seams(all: &[&Patch], kept: Vec<Kept>, tolerance: f64) -> Vec<Kept> {
    let mut out: Vec<Kept> = Vec::new();
    let mut by_patch: HashMap<usize, Vec<Kept>> = HashMap::new();
    for k in kept {
        if all[k.patch].wraps.iter().any(|&w| w) {
            by_patch.entry(k.patch).or_default().push(k);
        } else {
            out.push(k);
        }
    }
    let mut order: Vec<usize> = by_patch.keys().copied().collect();
    order.sort_unstable();
    for pi in order {
        let f = all[pi];
        let cells = by_patch.remove(&pi).unwrap();
        // Which seam a node is on: (direction, low side?).
        let side = |q: P2| -> Option<(usize, bool)> {
            (0..2).find_map(|k| {
                if !f.wraps[k] {
                    None
                } else if (q[k] - f.dlo[k]).abs() <= tolerance {
                    Some((k, true))
                } else if (q[k] - f.dhi[k]).abs() <= tolerance {
                    Some((k, false))
                } else {
                    None
                }
            })
        };
        // Seam sides, by the points they join.
        let mut low: HashMap<(usize, usize, usize), SideAt> = HashMap::new();
        let mut high: Vec<(usize, SideAt, (usize, usize))> = Vec::new();
        for (c, cell) in cells.iter().enumerate() {
            for (r, ring) in cell.rings.iter().enumerate() {
                let n = ring.len();
                for i in 0..n {
                    let (a, b) = (ring[i], ring[(i + 1) % n]);
                    match (side(a.uv), side(b.uv)) {
                        (Some((k, true)), Some((k2, true))) if k == k2 => {
                            low.insert((k, a.point, b.point), (c, r, i));
                        }
                        (Some((k, false)), Some((k2, false))) if k == k2 => {
                            high.push((k, (c, r, i), (a.point, b.point)));
                        }
                        _ => {}
                    }
                }
            }
        }
        // Offsets in whole periods, by a walk across matched seam sides. The
        // low side of the neighbour runs the other way.
        let mut links: Vec<Vec<Link>> = vec![Vec::new(); cells.len()];
        for &(k, mine, (pa, pb)) in &high {
            if let Some(&theirs) = low.get(&(k, pb, pa)) {
                links[mine.0].push((theirs.0, k, 1, mine, theirs));
                links[theirs.0].push((mine.0, k, -1, theirs, mine));
            }
        }
        let mut offset: Vec<Option<[i64; 2]>> = vec![None; cells.len()];
        let mut dropped: HashSet<SideAt> = HashSet::new();
        for start in 0..cells.len() {
            if offset[start].is_some() {
                continue;
            }
            offset[start] = Some([0, 0]);
            let mut queue = vec![start];
            while let Some(c) = queue.pop() {
                let oc = offset[c].unwrap();
                for &(d, k, step, mine, theirs) in &links[c] {
                    let mut od = oc;
                    od[k] += step;
                    match offset[d] {
                        None => {
                            offset[d] = Some(od);
                            queue.push(d);
                            dropped.insert(mine);
                            dropped.insert(theirs);
                        }
                        Some(x) if x == od => {
                            dropped.insert(mine);
                            dropped.insert(theirs);
                        }
                        Some(_) => {}
                    }
                }
            }
        }
        // Group the pieces that were joined.
        let mut group: Vec<usize> = (0..cells.len()).collect();
        fn find(g: &mut [usize], mut x: usize) -> usize {
            while g[x] != x {
                g[x] = g[g[x]];
                x = g[x];
            }
            x
        }
        for (c, list) in links.iter().enumerate() {
            for &(d, _, _, mine, _) in list {
                if dropped.contains(&mine) {
                    let (x, y) = (find(&mut group, c), find(&mut group, d));
                    group[x] = y;
                }
            }
        }
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for c in 0..cells.len() {
            let g = find(&mut group, c);
            members.entry(g).or_default().push(c);
        }
        let mut roots: Vec<usize> = members.keys().copied().collect();
        roots.sort_unstable();
        for g in roots {
            let list = &members[&g];
            let touched = list.iter().any(|&c| {
                cells[c]
                    .rings
                    .iter()
                    .enumerate()
                    .any(|(r, ring)| (0..ring.len()).any(|i| dropped.contains(&(c, r, i))))
            });
            if !touched {
                let c = list[0];
                out.push(Kept {
                    patch: pi,
                    flip: cells[c].flip,
                    rings: cells[c].rings.clone(),
                });
                continue;
            }
            // Directed sides, moved by their piece's offset, walked into
            // rings again.
            let mut sides: Vec<(Node, Node)> = Vec::new();
            for &c in list {
                let o = offset[c].unwrap();
                let shift = [
                    o[0] as f64 * f.period[0].unwrap_or(0.0),
                    o[1] as f64 * f.period[1].unwrap_or(0.0),
                ];
                let mv = |x: Node| Node {
                    point: x.point,
                    uv: [x.uv[0] + shift[0], x.uv[1] + shift[1]],
                };
                for (r, ring) in cells[c].rings.iter().enumerate() {
                    let n = ring.len();
                    for i in 0..n {
                        if !dropped.contains(&(c, r, i)) {
                            sides.push((mv(ring[i]), mv(ring[(i + 1) % n])));
                        }
                    }
                }
            }
            let mut used = vec![false; sides.len()];
            let mut rings: Vec<Vec<Node>> = Vec::new();
            for s in 0..sides.len() {
                if used[s] {
                    continue;
                }
                let mut ring = Vec::new();
                let mut at = s;
                loop {
                    used[at] = true;
                    ring.push(sides[at].0);
                    let end = sides[at].1;
                    let next = (0..sides.len()).find(|&t| {
                        !used[t]
                            && sides[t].0.point == end.point
                            && len2(sub2(sides[t].0.uv, end.uv)) <= tolerance * 4.0
                    });
                    match next {
                        Some(t) => at = t,
                        None => break,
                    }
                }
                if ring.len() >= 3 {
                    rings.push(ring);
                }
            }
            // The outline first.
            let area = |r: &Vec<Node>| signed_area(&r.iter().map(|n| n.uv).collect::<Vec<_>>());
            rings.sort_by(|x, y| area(y).total_cmp(&area(x)));
            out.push(Kept {
                patch: pi,
                flip: cells[list[0]].flip,
                rings,
            });
        }
    }
    out
}

/// The kept pieces as a body.
fn assemble(
    all: &[&Patch],
    kept: Vec<Kept>,
    pool: &Pool,
    origins: &Origins,
    tolerance: f64,
) -> Result<Body, Declined> {
    let pts = &pool.points;
    let used: Vec<usize> = {
        let mut u: Vec<usize> = kept
            .iter()
            .flat_map(|k| k.rings.iter().flatten().map(|n| n.point))
            .collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    // Every side split at every vertex lying on it — looked for in the cells
    // of a grid round the side, not among every vertex there is.
    let cell = {
        let mut lengths: Vec<f64> = kept
            .iter()
            .flat_map(|k| k.rings.iter())
            .flat_map(|r| {
                r.windows(2)
                    .map(|w| v3::dist(pts[w[0].point], pts[w[1].point]))
            })
            .collect();
        lengths.sort_by(f64::total_cmp);
        lengths
            .get(lengths.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(tolerance * 4.0)
            .max(1e-12)
    };
    let key = |p: V3| p.map(|x| (x / cell).floor() as i64);
    let mut filed: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for &v in &used {
        filed.entry(key(pts[v])).or_default().push(v);
    }
    let near = |pa: V3, pb: V3| -> Vec<usize> {
        let lo = key([0, 1, 2].map(|k| pa[k].min(pb[k]) - tolerance));
        let hi = key([0, 1, 2].map(|k| pa[k].max(pb[k]) + tolerance));
        let count: i64 = (0..3).map(|k| hi[k] - lo[k] + 1).product();
        if count > 4096 {
            return used.clone();
        }
        let mut out = Vec::new();
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    if let Some(list) = filed.get(&[x, y, z]) {
                        out.extend(list);
                    }
                }
            }
        }
        out
    };
    let split = |ring: &[Node]| -> Vec<Node> {
        let n = ring.len();
        let mut out: Vec<Node> = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            out.push(a);
            let (pa, pb) = (pts[a.point], pts[b.point]);
            let ab = v3::sub(pb, pa);
            let l = v3::norm(ab);
            if l <= tolerance {
                continue;
            }
            let mut on: Vec<(f64, usize)> = near(pa, pb)
                .iter()
                .filter(|&&v| v != a.point && v != b.point)
                .filter_map(|&v| {
                    let t = v3::dot(v3::sub(pts[v], pa), ab) / (l * l);
                    if t * l <= tolerance || (1.0 - t) * l <= tolerance {
                        return None;
                    }
                    (v3::dist(lerp3(pa, pb, t), pts[v]) <= tolerance).then_some((t, v))
                })
                .collect();
            on.sort_by(|x, y| x.0.total_cmp(&y.0));
            out.extend(on.into_iter().map(|(t, v)| Node {
                point: v,
                uv: [
                    a.uv[0] + (b.uv[0] - a.uv[0]) * t,
                    a.uv[1] + (b.uv[1] - a.uv[1]) * t,
                ],
            }));
        }
        out
    };

    struct Built {
        patch: usize,
        flipped: bool,
        rings: Vec<Vec<Node>>,
        /// A whole band round its surface with nothing cut out of it: its
        /// seam sides, which it does not keep. It is a face that wraps, as
        /// the primitives make one, bounded by its rims alone.
        band: Option<HashSet<usize>>,
    }
    let mut built: Vec<Built> = Vec::new();
    for k in kept {
        let rings: Vec<Vec<Node>> = k
            .rings
            .iter()
            .map(|r| split(r))
            .filter(|r| r.len() >= 3)
            .collect();
        if rings.is_empty() {
            continue;
        }
        let f = all[k.patch];
        let band = match (f.wraps, f.period[0]) {
            ([true, false], Some(t)) if rings.len() == 1 => {
                let ring = &rings[0];
                let n = ring.len();
                let on_seam = |q: P2| {
                    let k = (q[0] - f.dlo[0]) / t;
                    (k - k.round()).abs() * t <= tolerance
                };
                let seam: HashSet<usize> = (0..n)
                    .filter(|&i| {
                        let (a, b) = (ring[i].uv, ring[(i + 1) % n].uv);
                        on_seam(a) && on_seam(b) && (a[0] - b[0]).abs() <= tolerance
                    })
                    .collect();
                let (lo, hi) = ring
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |m, x| {
                        (m.0.min(x.uv[0]), m.1.max(x.uv[0]))
                    });
                // And the rest is rims: each at one height, the lowest or the
                // highest, since a face that wraps is read as the band
                // between them. A sphere with a square column through it
                // goes all the way round, and its edges are not rims.
                let (vlo, vhi) = ring
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |m, x| {
                        (m.0.min(x.uv[1]), m.1.max(x.uv[1]))
                    });
                let rims = ring.iter().all(|x| {
                    (x.uv[1] - vlo).abs() <= tolerance || (x.uv[1] - vhi).abs() <= tolerance
                });
                ((hi - lo - t).abs() <= tolerance && !seam.is_empty() && rims).then_some(seam)
            }
            _ => None,
        };
        built.push(Built {
            patch: k.patch,
            flipped: f.flipped ^ k.flip,
            rings,
            band,
        });
    }

    // A band whose rim the face beside it has cut in two — at the end of
    // that face's own seam — is not read back right as a face that wraps,
    // so it keeps its seam instead, and the edges are worked out again.
    let chains = loop {
        let chains = {
            // Sides by the points they join, with the faces along each.
            let mut uses: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for (fi, b) in built.iter().enumerate() {
                for (r, ring) in b.rings.iter().enumerate() {
                    let n = ring.len();
                    for i in 0..n {
                        if r == 0 && b.band.as_ref().is_some_and(|s| s.contains(&i)) {
                            continue;
                        }
                        let (p, q) = (ring[i].point, ring[(i + 1) % n].point);
                        if p != q {
                            uses.entry((p.min(q), p.max(q))).or_default().push(fi);
                        }
                    }
                }
            }
            let open = uses.values().filter(|u| u.len() != 2).count();
            if open > 0 {
                if std::env::var_os("THREERS_BREP_DEBUG").is_some() {
                    for (fi, b) in built.iter().enumerate() {
                        let zs: Vec<f64> =
                            b.rings.iter().flatten().map(|n| pts[n.point][2]).collect();
                        let (lo, hi) =
                            zs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &z| {
                                (a.0.min(z), a.1.max(z))
                            });
                        eprintln!(
                            "  [fuse] face {fi}: patch {} {} rings {} z {lo}..{hi}",
                            b.patch,
                            all[b.patch].surface.kind(),
                            b.rings.len()
                        );
                    }
                    for (k, u) in &uses {
                        if u.len() != 2 {
                            let kinds: Vec<&str> = u
                                .iter()
                                .map(|&f| all[built[f].patch].surface.kind())
                                .collect();
                            eprintln!(
                                "  [fuse] side {:?} .. {:?} faces {u:?} {kinds:?}",
                                pts[k.0], pts[k.1]
                            );
                        }
                    }
                }
                return Err(Declined::NotWatertight { open_edges: open });
            }

            // Chain sides into edges through points where nothing else meets.
            let mut at: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for &key in uses.keys() {
                at.entry(key.0).or_default().push(key);
                at.entry(key.1).or_default().push(key);
            }
            let pair = |key: &(usize, usize)| {
                let mut u = uses[key].clone();
                u.sort_unstable();
                u
            };
            // Through a point where nothing else meets and the line does
            // not turn a corner: the same two faces can meet along a line
            // and then an arc, and those are two edges with a vertex
            // between. A corner is a turn no sampled curve makes — a circle
            // has twelve points at the least, 30° apart — since a coarse rim
            // of 44 points turns 8° at every one.
            let through = |v: usize| -> bool {
                let list = &at[&v];
                if list.len() != 2 || pair(&list[0]) != pair(&list[1]) {
                    return false;
                }
                let other = |k: &(usize, usize)| if k.0 == v { k.1 } else { k.0 };
                let (a, b) = (
                    v3::sub(pts[v], pts[other(&list[0])]),
                    v3::sub(pts[other(&list[1])], pts[v]),
                );
                let (la, lb) = (v3::norm(a), v3::norm(b));
                la <= 0.0 || lb <= 0.0 || v3::dot(a, b) / (la * lb) >= (35.0f64).to_radians().cos()
            };
            let mut keys: Vec<(usize, usize)> = uses.keys().copied().collect();
            keys.sort_unstable();
            let mut chain_of: HashMap<(usize, usize), usize> = HashMap::new();
            let mut chains: Vec<(Vec<usize>, Vec<usize>)> = Vec::new(); // points, faces
            let walk = |from: usize,
                        mut via: (usize, usize),
                        id: usize,
                        chain_of: &mut HashMap<(usize, usize), usize>| {
                let mut out = Vec::new();
                let mut v = from;
                loop {
                    let next_v = if via.0 == v { via.1 } else { via.0 };
                    out.push(next_v);
                    if !through(next_v) {
                        break;
                    }
                    let Some(&next) = at[&next_v].iter().find(|&&k| k != via) else {
                        break;
                    };
                    if chain_of.contains_key(&next) {
                        break;
                    }
                    chain_of.insert(next, id);
                    via = next;
                    v = next_v;
                }
                out
            };
            for &key in &keys {
                if chain_of.contains_key(&key) {
                    continue;
                }
                let id = chains.len();
                chain_of.insert(key, id);
                let mut points = vec![key.0];
                points.extend(walk(key.0, key, id, &mut chain_of));
                let closed = points.last() == Some(&key.0);
                if !closed
                    && through(key.0)
                    && let Some(start) = at[&key.0].iter().copied().find(|&k| k != key)
                    && !chain_of.contains_key(&start)
                {
                    chain_of.insert(start, id);
                    let mut back = walk(key.0, start, id, &mut chain_of);
                    back.reverse();
                    back.extend(points);
                    points = back;
                }
                chains.push((points, pair(&key)));
            }
            chains
        };
        let mut changed = false;
        for (fi, b) in built.iter_mut().enumerate() {
            if b.band.is_some()
                && chains
                    .iter()
                    .any(|(pts, users)| users.contains(&fi) && pts.first() != pts.last())
            {
                b.band = None;
                changed = true;
            }
        }
        if !changed {
            break chains;
        }
    };
    let mut index: HashMap<usize, usize> = HashMap::new();
    let mut vertices: Vec<V3> = Vec::new();
    let mut name = |v: usize, vertices: &mut Vec<V3>| {
        *index.entry(v).or_insert_with(|| {
            vertices.push(pts[v]);
            vertices.len() - 1
        })
    };
    // One surface per face of the inputs, however many pieces it became, so
    // a point is on as many surfaces as there are and no more.
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut surface_of: HashMap<usize, usize> = HashMap::new();
    let mut faces = Vec::with_capacity(built.len());
    for b in &built {
        let f = all[b.patch];
        let surface = *surface_of.entry(b.patch).or_insert_with(|| {
            surfaces.push(f.surface.clone());
            surfaces.len() - 1
        });
        let mut loops = Vec::with_capacity(b.rings.len());
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for ring in &b.rings {
            let uv: Vec<P2> = ring
                .iter()
                .map(|n| [n.uv[0] / f.scale[0], n.uv[1] / f.scale[1]])
                .collect();
            for q in &uv {
                for k in 0..2 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            loops.push(TrimLoop {
                area: signed_area(&uv),
                uv,
                vertices: ring.iter().map(|n| name(n.point, &mut vertices)).collect(),
            });
        }
        let band = b.band.is_some();
        faces.push(Face {
            surface,
            edges: Vec::new(),
            flipped: b.flipped,
            u_range: (lo[0], hi[0]),
            v_range: (lo[1], hi[1]),
            u_wraps: band,
            v_wraps: false,
            loops: (!band).then_some(loops),
        });
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(chains.len());
    for (points, users) in &chains {
        let closed = points.len() > 2 && points.first() == points.last();
        let vs: Vec<usize> = points.iter().map(|&v| name(v, &mut vertices)).collect();
        edges.push(Edge {
            surfaces: (faces[users[0]].surface, faces[users[1]].surface),
            vertices: vs,
            closed,
        });
        let e = edges.len() - 1;
        for &fi in users {
            faces[fi].edges.push(e);
        }
    }
    // Every point on every surface it is a point of.
    settle(
        &surfaces,
        &mut vertices,
        &mut faces,
        &edges,
        origins,
        tolerance,
    );
    merge_coincident(&vertices, &mut faces, &mut edges, tolerance);
    bend(
        &surfaces,
        &mut vertices,
        &mut faces,
        &mut edges,
        origins,
        tolerance,
    );
    Ok(Body::from_parts(surfaces, vertices, edges, faces))
}

/// A middle point for every edge of two points that is not straight.
///
/// Where curves cross close together, a piece of an arc between two of the
/// crossings can come out with no sample in it: two points, which read back
/// as a straight line, and which piece does depends on where the sampling
/// fell. Its midpoint, put on the curve — the circle it was drawn from, or
/// where its two surfaces meet — says what it is.
fn bend(
    surfaces: &[Surface],
    vertices: &mut Vec<V3>,
    faces: &mut [Face],
    edges: &mut [Edge],
    origins: &Origins,
    tolerance: f64,
) {
    let tiny = tolerance * 1e-6;
    let mut added: Vec<(usize, usize, usize)> = Vec::new(); // ends and middle
    for e in edges.iter_mut() {
        if e.vertices.len() != 2 || e.closed {
            continue;
        }
        let (a, b) = (e.vertices[0], e.vertices[1]);
        let (pa, pb) = (vertices[a], vertices[b]);
        let m = lerp3(pa, pb, 0.5);
        // The circle of an input edge it runs along: both ends on it, and
        // its middle by that edge's own polyline — a straight cut across a
        // disc has both ends on the rim too.
        let circle = origins
            .at(pa, tolerance)
            .into_iter()
            .filter_map(|(_, _, c)| c)
            .find(|&c| {
                let reach = tolerance.max(origins.circles[c].3 * 2.0);
                origins.off_circle(c, pa) <= tiny
                    && origins.off_circle(c, pb) <= tiny
                    && origins.at(m, reach).iter().any(|x| x.2 == Some(c))
            });
        let q = match circle {
            Some(c) => {
                // The nearer of the two arcs between the ends: the point out
                // from the centre through the chord's middle.
                origins.onto_circle(c, m)
            }
            None if e.surfaces.0 != e.surfaces.1 => {
                match super::intersect::settle(
                    &surfaces[e.surfaces.0],
                    &surfaces[e.surfaces.1],
                    m,
                    tolerance,
                ) {
                    Some(q) if v3::dist(q, m) <= v3::dist(pa, pb) => q,
                    _ => continue,
                }
            }
            None => continue,
        };
        if v3::dist(q, m) <= tiny.max(1e-12) {
            continue;
        }
        vertices.push(q);
        let mid = vertices.len() - 1;
        e.vertices = vec![a, mid, b];
        added.push((a, b, mid));
    }
    if added.is_empty() {
        return;
    }
    // And into the loops that step from one end to the other.
    let between: HashMap<(usize, usize), usize> = added
        .iter()
        .flat_map(|&(a, b, m)| [((a, b), m), ((b, a), m)])
        .collect();
    for f in faces.iter_mut() {
        let s = &surfaces[f.surface];
        for l in f.loops.iter_mut().flatten() {
            let n = l.vertices.len();
            let mut vs = Vec::with_capacity(n + 2);
            let mut uv = Vec::with_capacity(n + 2);
            for k in 0..n {
                let (a, b) = (l.vertices[k], l.vertices[(k + 1) % n]);
                vs.push(a);
                uv.push(l.uv[k]);
                if let Some(&m) = between.get(&(a, b)) {
                    let near = [
                        (l.uv[k][0] + l.uv[(k + 1) % n][0]) * 0.5,
                        (l.uv[k][1] + l.uv[(k + 1) % n][1]) * 0.5,
                    ];
                    let q = s
                        .invert_near(vertices[m], (near[0], near[1]))
                        .map(|(u, w)| [u, w])
                        .unwrap_or(near);
                    vs.push(m);
                    uv.push(q);
                }
            }
            l.vertices = vs;
            l.uv = uv;
        }
    }
}

/// The edges of the inputs, as segments with the two surfaces each lies
/// between.
struct Origins {
    surfaces: Vec<Surface>,
    /// Ends, the two surfaces, and the circle the whole edge lies on if
    /// it lies on one.
    segments: Vec<(V3, V3, usize, usize, Option<usize>)>,
    /// Centre, normal, radius, and how far a chord of the edge sags.
    circles: Vec<(V3, V3, f64, f64)>,
    grid: HashMap<[i64; 3], Vec<usize>>,
    /// Segments too long to file by cell, checked every time.
    long: Vec<usize>,
    cell: f64,
}

impl Origins {
    fn of(bodies: &[&Body], tolerance: f64) -> Origins {
        let mut surfaces = Vec::new();
        let mut segments = Vec::new();
        let mut circles = Vec::new();
        for b in bodies {
            let base = surfaces.len();
            surfaces.extend(b.surfaces().iter().cloned());
            for e in b.edges() {
                // An edge between a surface and itself — a seam, or the
                // line between two pieces of one plane — adds no surface,
                // but its circle, if it is one, is still its curve.
                let pts: Vec<V3> = e.vertices.iter().map(|&v| b.vertices()[v]).collect();
                // An edge between two pieces of one plane is a polyline the
                // surfaces cannot place a point on; if it samples a circle,
                // the circle can.
                if std::env::var_os("THREERS_BREP_DEBUG").is_some()
                    && pts.len() >= 4
                    && circle_through(&pts).is_none()
                {
                    let (a, z) = (pts[0], pts[pts.len() - 1]);
                    let bent = pts
                        .iter()
                        .map(|p| to_polyline(*p, &[a, z]))
                        .fold(0.0, f64::max);
                    if bent > tolerance {
                        eprintln!(
                            "  [fuse] input edge of {} points bent {bent:.2e}, neither straight nor a circle, between {} and {}",
                            pts.len(),
                            b.surfaces()[e.surfaces.0].kind(),
                            b.surfaces()[e.surfaces.1].kind()
                        );
                    }
                }
                let circle = circle_through(&pts).map(|c| {
                    circles.push(c);
                    circles.len() - 1
                });
                for w in pts.windows(2) {
                    segments.push((w[0], w[1], base + e.surfaces.0, base + e.surfaces.1, circle));
                }
            }
        }
        // Cells about as big as the segments are long, mostly.
        let mut lengths: Vec<f64> = segments.iter().map(|s| v3::dist(s.0, s.1)).collect();
        lengths.sort_by(f64::total_cmp);
        let cell = lengths
            .get(lengths.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(tolerance * 8.0);
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        let mut long = Vec::new();
        for (i, (p, q, _, _, _)) in segments.iter().enumerate() {
            let k = |x: f64| (x / cell).floor() as i64;
            let lo = [0, 1, 2].map(|d| k(p[d].min(q[d]) - tolerance));
            let hi = [0, 1, 2].map(|d| k(p[d].max(q[d]) + tolerance));
            if (0..3)
                .map(|d| (hi[d] - lo[d] + 1) as i128)
                .product::<i128>()
                > 64
            {
                long.push(i);
                continue;
            }
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        grid.entry([x, y, z]).or_default().push(i);
                    }
                }
            }
        }
        Origins {
            surfaces,
            segments,
            circles,
            grid,
            long,
            cell,
        }
    }

    /// The surface pairs, and circles, of the input edges passing within
    /// `tolerance` of `p`.
    fn at(&self, p: V3, reach: f64) -> Vec<(usize, usize, Option<usize>)> {
        let tolerance = reach;
        let k = p.map(|x| (x / self.cell).floor() as i64);
        let mut out = Vec::new();
        for &i in self.grid.get(&k).into_iter().flatten().chain(&self.long) {
            let (a, b, s, t, c) = self.segments[i];
            if to_polyline(p, &[a, b]) <= tolerance && !out.contains(&(s, t, c)) {
                out.push((s, t, c));
            }
        }
        out
    }
}

/// The circle a polyline's points all lie on, to rounding: centre, normal,
/// radius, and the sag of its longest chord. `None` for a straight one, or
/// one that is not a circle.
fn circle_through(pts: &[V3]) -> Option<(V3, V3, f64, f64)> {
    let n = pts.len();
    if n < 4 {
        return None;
    }
    let (a, m, b) = (pts[0], pts[n / 3], pts[2 * n / 3]);
    let (u, w) = (v3::sub(m, a), v3::sub(b, a));
    let uxw = v3::cross(u, w);
    let l2 = v3::dot(uxw, uxw);
    if l2 <= 1e-24 {
        return None;
    }
    let t = v3::cross(
        v3::sub(v3::scale(w, v3::dot(u, u)), v3::scale(u, v3::dot(w, w))),
        uxw,
    );
    let centre = v3::add(a, v3::scale(t, 0.5 / l2));
    let r = v3::dist(a, centre);
    let normal = v3::normalize(uxw)?;
    let eps = 1e-9 * r.max(1.0);
    let on = pts.iter().all(|p| {
        let d = v3::sub(*p, centre);
        (v3::norm(d) - r).abs() <= eps && v3::dot(d, normal).abs() <= eps
    });
    if !on {
        return None;
    }
    let chord = pts
        .windows(2)
        .map(|w| v3::dist(w[0], w[1]))
        .fold(0.0, f64::max);
    Some((
        centre,
        normal,
        r,
        r - (r * r - chord * chord / 4.0).max(0.0).sqrt(),
    ))
}

impl Origins {
    /// `p` moved onto a circle, in its plane, out from its centre.
    fn onto_circle(&self, c: usize, p: V3) -> V3 {
        let (centre, normal, r, _) = self.circles[c];
        let d = v3::sub(p, centre);
        let flat = v3::sub(d, v3::scale(normal, v3::dot(d, normal)));
        match v3::normalize(flat) {
            Some(dir) => v3::add(centre, v3::scale(dir, r)),
            None => p,
        }
    }

    /// Where a plane crossing a circle's own plane meets the circle, nearest
    /// `p`; `None` for a plane parallel to it.
    fn through_plane(&self, c: usize, origin: V3, normal: V3, p: V3) -> Option<V3> {
        let (centre, n, r, _) = self.circles[c];
        let dir = v3::cross(n, normal);
        let l = v3::norm(dir);
        if l <= 1e-6 {
            return None;
        }
        let dir = v3::scale(dir, 1.0 / l);
        // The point of the line nearest the centre: in the circle's plane,
        // on the other plane, square to the line.
        let across = v3::cross(dir, n);
        let k = v3::dot(normal, v3::sub(origin, centre)) / v3::dot(normal, across);
        if !k.is_finite() || k.abs() > r {
            return None;
        }
        let foot = v3::add(centre, v3::scale(across, k));
        let half = (r * r - k * k).max(0.0).sqrt();
        [1.0, -1.0]
            .map(|s| v3::add(foot, v3::scale(dir, s * half)))
            .into_iter()
            .min_by(|u, w| v3::dist(*u, p).total_cmp(&v3::dist(*w, p)))
    }

    /// Where two circles in one plane cross, nearest `p`.
    fn crossing(&self, a: usize, b: usize, p: V3) -> Option<V3> {
        let (ca, na, ra, _) = self.circles[a];
        let (cb, nb, rb, _) = self.circles[b];
        let scale = ra.max(rb).max(1.0);
        if v3::norm(v3::cross(na, nb)) > 1e-9 || v3::dot(na, v3::sub(cb, ca)).abs() > 1e-9 * scale {
            return None;
        }
        let d = v3::sub(cb, ca);
        let dist = v3::norm(d);
        if dist <= 1e-12 * scale || dist > ra + rb || dist < (ra - rb).abs() {
            return None;
        }
        let x = v3::scale(d, 1.0 / dist);
        let y = v3::cross(na, x);
        // Along the line of centres, then either way across it.
        let along = (dist * dist + ra * ra - rb * rb) / (2.0 * dist);
        let across = (ra * ra - along * along).max(0.0).sqrt();
        let foot = v3::add(ca, v3::scale(x, along));
        [1.0, -1.0]
            .map(|s| v3::add(foot, v3::scale(y, s * across)))
            .into_iter()
            .min_by(|u, w| v3::dist(*u, p).total_cmp(&v3::dist(*w, p)))
    }

    fn off_circle(&self, c: usize, p: V3) -> f64 {
        v3::dist(p, self.onto_circle(c, p))
    }
}

/// Where a point claimed by a plane, a cylinder tangent to it within the
/// tolerance, and one more plane is: on the line the two touch along, where
/// it crosses the other plane. `None` where the claims are not that.
fn tangent_point(surfaces: &[&Surface], claims: &[usize], p: V3, tolerance: f64) -> Option<V3> {
    if claims.len() != 3 {
        return None;
    }
    for &c in claims {
        let Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } = surfaces[c]
        else {
            continue;
        };
        let planes: Vec<(V3, V3)> = claims
            .iter()
            .filter(|&&s| s != c)
            .filter_map(|&s| match surfaces[s] {
                Surface::Plane { origin, normal, .. } => Some((*origin, *normal)),
                _ => None,
            })
            .collect();
        if planes.len() != 2 {
            continue;
        }
        for (k, &(po, pn)) in planes.iter().enumerate() {
            // Along the axis, and as far from it as the radius.
            if v3::dot(pn, *axis).abs() > 1e-9 {
                continue;
            }
            let gap = v3::dot(pn, v3::sub(*origin, po));
            if (gap.abs() - radius).abs() > tolerance {
                continue;
            }
            // The line of the cylinder nearest the plane.
            let toward = if gap > 0.0 { v3::scale(pn, -1.0) } else { pn };
            let base = v3::add(*origin, v3::scale(toward, *radius));
            // Where it crosses the other plane.
            let (qo, qn) = planes[1 - k];
            let along = v3::dot(qn, *axis);
            if along.abs() <= 1e-9 {
                return None;
            }
            let t = v3::dot(qn, v3::sub(qo, base)) / along;
            let q = v3::add(base, v3::scale(*axis, t));
            return (v3::dist(p, q) <= tolerance * 1e3).then_some(q);
        }
    }
    None
}

/// Vertices that settled onto one point are one vertex: every reference to
/// the later turned to the earlier, a ring's step from one to itself
/// dropped, and an edge that shrank to a point dropped with it.
fn merge_coincident(vertices: &[V3], faces: &mut [Face], edges: &mut Vec<Edge>, tolerance: f64) {
    let near = tolerance * 0.1;
    let key = |p: V3| p.map(|x| (x / near).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut onto: Vec<usize> = (0..vertices.len()).collect();
    for (i, p) in vertices.iter().enumerate() {
        let k = key(*p);
        let mut found = None;
        'look: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid
                        .get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                        .into_iter()
                        .flatten()
                    {
                        if v3::dist(vertices[j], *p) <= near {
                            found = Some(j);
                            break 'look;
                        }
                    }
                }
            }
        }
        match found {
            Some(j) => onto[i] = j,
            None => grid.entry(k).or_default().push(i),
        }
    }
    if onto.iter().enumerate().all(|(i, &j)| i == j) {
        return;
    }
    let mut gone = vec![false; edges.len()];
    for (i, e) in edges.iter_mut().enumerate() {
        for v in &mut e.vertices {
            *v = onto[*v];
        }
        e.vertices.dedup();
        if e.vertices.len() < 2 || (e.vertices.len() == 2 && e.closed) {
            gone[i] = true;
        }
    }
    for f in faces.iter_mut() {
        f.edges.retain(|&e| !gone[e]);
        for l in f.loops.iter_mut().flatten() {
            let mut vs = Vec::with_capacity(l.vertices.len());
            let mut uv = Vec::with_capacity(l.uv.len());
            for (k, &v) in l.vertices.iter().enumerate() {
                let v = onto[v];
                if vs.last() != Some(&v) {
                    vs.push(v);
                    uv.push(l.uv[k]);
                }
            }
            while vs.len() > 1 && vs.first() == vs.last() {
                vs.pop();
                uv.pop();
            }
            l.vertices = vs;
            l.uv = uv;
        }
    }
    // Edges no face names any more go, and the rest are renumbered.
    let mut index = vec![usize::MAX; edges.len()];
    let mut kept = Vec::with_capacity(edges.len());
    for (i, e) in edges.drain(..).enumerate() {
        if !gone[i] {
            index[i] = kept.len();
            kept.push(e);
        }
    }
    *edges = kept;
    for f in faces.iter_mut() {
        for e in &mut f.edges {
            *e = index[*e];
        }
    }
}

/// Put each vertex exactly on every surface it belongs to.
///
/// A run's end is found by halving along a chord, a point where one edge
/// meets another is put on a chord, a crossing is evaluated on one of the two
/// surfaces: each is within the tolerance and no closer. An edge whose
/// points are not on its curve is not that curve — read back, a circle with
/// one such point is no circle. Two surfaces settle by Newton on the pair;
/// three or more, at a corner, by projecting onto each in turn.
fn settle(
    surfaces: &[Surface],
    vertices: &mut [V3],
    faces: &mut [Face],
    edges: &[Edge],
    origins: &Origins,
    tolerance: f64,
) {
    let mut claims: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut claim = |v: usize, s: usize| {
        let c = claims.entry(v).or_default();
        if !c.contains(&s) {
            c.push(s);
        }
    };
    for f in faces.iter() {
        for l in f.loops.iter().flatten() {
            for &v in &l.vertices {
                claim(v, f.surface);
            }
        }
        for &e in &f.edges {
            for &v in &edges[e].vertices {
                claim(v, f.surface);
            }
        }
    }
    // And the curve of any edge of the inputs a vertex lies on: a solid's
    // outline laid on a face of the other lies between two pieces of one
    // plane, which say nothing about where along it a point is; the edge it
    // came from does.
    let mut on_circles: HashMap<usize, Vec<usize>> = HashMap::new();
    for (v, c) in claims.iter_mut() {
        // On the edge to rounding, as a sample of it or a point found on
        // one of its chords is: near it within the tolerance is not enough,
        // since a point that close to a circle need not be on it.
        for (a, b, circle) in origins.at(vertices[*v], tolerance * 1e-5) {
            for s in [a, b] {
                let s = surfaces.len() + s;
                if !c.contains(&s) {
                    c.push(s);
                }
            }
            if let Some(k) = circle {
                let list = on_circles.entry(*v).or_default();
                if !list.contains(&k) {
                    list.push(k);
                }
            }
        }
    }
    // The ends of an edge whose inner points are all on one circle are on it
    // too: they are where it meets the next curve, found within the
    // tolerance of a chord rather than on it.
    for e in edges {
        let n = e.vertices.len();
        if n < 2 {
            continue;
        }
        let common: Vec<usize> = if n == 2 {
            // No inner point: the circle of an input edge its middle runs
            // along.
            let m = lerp3(vertices[e.vertices[0]], vertices[e.vertices[1]], 0.5);
            origins
                .at(m, tolerance)
                .into_iter()
                .filter_map(|x| x.2)
                .filter(|&c| {
                    origins
                        .at(m, tolerance.max(origins.circles[c].3 * 2.0))
                        .iter()
                        .any(|x| x.2 == Some(c))
                })
                .collect()
        } else {
            let inner = &e.vertices[1..n - 1];
            on_circles
                .get(&inner[0])
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|c| {
                    inner
                        .iter()
                        .all(|v| on_circles.get(v).is_some_and(|l| l.contains(c)))
                })
                .collect()
        };
        for &end in [e.vertices[0], e.vertices[n - 1]].iter() {
            for &c in &common {
                // Within the sag of the circle's chords: a crossing found on
                // a chord is that far off it.
                if origins.off_circle(c, vertices[end]) <= tolerance.max(origins.circles[c].3 * 2.0)
                {
                    let list = on_circles.entry(end).or_default();
                    if !list.contains(&c) {
                        list.push(c);
                    }
                    claims.entry(end).or_default();
                }
            }
        }
    }
    let result = surfaces;
    let surfaces: Vec<&Surface> = result.iter().chain(origins.surfaces.iter()).collect();
    let worst = |p: V3, ss: &[usize]| {
        ss.iter()
            .map(|&s| surfaces[s].distance(p).abs())
            .fold(0.0, f64::max)
    };
    let probe: Option<V3> = std::env::var("FUSE_PROBE").ok().and_then(|s| {
        let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| [v[0], v[1], v[2]])
    });
    let mut moved: HashMap<usize, V3> = HashMap::new();
    for (&v, ss) in &claims {
        let p = vertices[v];
        let cs: &[usize] = on_circles.get(&v).map(|c| c.as_slice()).unwrap_or(&[]);
        if probe.is_some_and(|x| v3::dist(x, p) < 0.05) {
            eprintln!(
                "  [probe] vertex {v} at {p:?}: surfaces {:?} circles {:?} near origins {:?}",
                ss.iter().map(|&s| surfaces[s].kind()).collect::<Vec<_>>(),
                cs.iter().map(|&c| origins.circles[c]).collect::<Vec<_>>(),
                origins.at(p, tolerance)
            );
        }
        let worst = |q: V3| {
            cs.iter()
                .map(|&c| origins.off_circle(c, q))
                .fold(worst(q, ss), f64::max)
        };
        let was = worst(p);
        if ss.len() + cs.len() < 2 || was <= 1e-12 {
            continue;
        }
        // A plane and a cylinder lying along it that touch within the
        // tolerance touch along one line of the cylinder: the one nearest the
        // plane. A point on both is on that line; with a third plane, where
        // the line crosses it. Taking turns onto the two crawls along the gap
        // between them and stops a tolerance short of where they touch — a
        // pocket floor's side came back bent there.
        let tangent = cs
            .is_empty()
            .then(|| tangent_point(&surfaces, ss, p, tolerance))
            .flatten();
        let q = if let Some(q) = tangent {
            q
        } else if ss.len() == 2 && cs.is_empty() {
            match super::intersect::settle(surfaces[ss[0]], surfaces[ss[1]], p, tolerance) {
                Some(q) => q,
                None => continue,
            }
        } else if let Some(q) = (cs.len() >= 2)
            .then(|| origins.crossing(cs[0], cs[1], p))
            .flatten()
            .or_else(|| {
                // A circle and a plane across its own: where the line they
                // share meets it.
                let c = *cs.first()?;
                ss.iter().find_map(|&s| match surfaces[s] {
                    Surface::Plane { origin, normal, .. } => {
                        origins.through_plane(c, *origin, *normal, p)
                    }
                    _ => None,
                })
            })
        {
            // Two circles in one plane cross where a formula says: taking
            // turns onto each crawls where they meet at a shallow angle, and
            // stops short of it.
            q
        } else {
            // Onto each surface and each circle in turn, until on all.
            let mut q = p;
            for _ in 0..200 {
                for &s in ss {
                    let surface = surfaces[s];
                    if let Some((u, w)) = surface.invert(q) {
                        q = surface.point(u, w);
                    }
                }
                for &c in cs {
                    q = origins.onto_circle(c, q);
                }
                if worst(q) <= 1e-12 {
                    break;
                }
            }
            q
        };
        // As far as the tolerance, or as far as a chord of the circle sags:
        // a point found on the chord is that far from the curve.
        let reach = cs
            .iter()
            .map(|&c| origins.circles[c].3 * 2.0)
            .fold(tolerance * 4.0, f64::max);
        if probe.is_some_and(|x| v3::dist(x, p) < 0.05) {
            eprintln!(
                "  [probe] -> {q:?}: moved {:.3e} of {reach:.3e}, worst {:.3e} was {was:.3e}; per surface {:?}",
                v3::dist(p, q),
                worst(q),
                ss.iter().map(|&s| surfaces[s].distance(q)).collect::<Vec<_>>()
            );
        }
        // Further, where the point it lands on is exactly on everything it
        // belongs to: a line meeting an arc at a shallow angle crosses the
        // chord well along from the arc — half a millimetre for a 0.01 sag
        // at about a degree.
        // A tangency within the tolerance is further still: along the line
        // they touch, a curve within the tolerance of it runs the square
        // root of the gap — a fifth of a millimetre for a thousandth.
        let exact = worst(q) <= tolerance * 1e-9;
        let far = if tangent.is_some() {
            tolerance * 1e3
        } else if exact {
            reach * 50.0
        } else {
            reach
        };
        if v3::dist(p, q) <= far && worst(q) < was {
            moved.insert(v, q);
        }
        if std::env::var_os("THREERS_BREP_DEBUG").is_some() && worst(q) > 1e-9 {
            eprintln!(
                "  [fuse] vertex {v} unsettled: off by {:.2e} (was {was:.2e}), moved {:.2e} of {reach:.2e}, {} surfaces {:?} {} circles",
                worst(q),
                v3::dist(p, q),
                ss.len(),
                ss.iter().map(|&s| surfaces[s].kind()).collect::<Vec<_>>(),
                cs.len()
            );
        }
    }
    for (v, q) in &moved {
        vertices[*v] = *q;
    }
    for f in faces.iter_mut() {
        let s = surfaces[f.surface];
        let (pu, pv) = s.periodic();
        for l in f.loops.iter_mut().flatten() {
            for k in 0..l.vertices.len() {
                let Some(q) = moved.get(&l.vertices[k]) else {
                    continue;
                };
                let Some((u, w)) = s.invert(*q) else {
                    continue;
                };
                let old = l.uv[k];
                let near = |x: f64, a: f64| {
                    x - std::f64::consts::TAU * ((x - a) / std::f64::consts::TAU).round()
                };
                l.uv[k] = [
                    if pu { near(u, old[0]) } else { u },
                    if pv { near(w, old[1]) } else { w },
                ];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Volume of a body bounded by planes, from its loops: each face's area
    /// times its plane's distance from the origin, over three.
    fn volume(body: &Body) -> f64 {
        body.faces()
            .iter()
            .map(|f| {
                let Surface::Plane { origin, normal, .. } = &body.surfaces()[f.surface] else {
                    panic!("not flat");
                };
                let area: f64 = f.loops.iter().flatten().map(|l| l.area).sum();
                let sign = if f.flipped { -1.0 } else { 1.0 };
                sign * v3::dot(*normal, *origin) * area / 3.0
            })
            .sum()
    }

    fn check(a: &Body, b: &Body, op: BooleanOp, expected: f64) {
        let r = planar(a, b, op, 1e-6).unwrap().unwrap();
        assert!(
            r.faces().is_empty() || r.is_valid_solid(1e-6),
            "{op:?}: {:?}",
            r.defects(1e-6)
        );
        let v = volume(&r);
        assert!(
            (v - expected).abs() < 1e-9,
            "{op:?}: volume {v}, expected {expected}"
        );
    }

    #[test]
    fn two_boxes_overlapping_at_a_corner() {
        let a = Body::cuboid([4.0, 4.0, 4.0]);
        let b = Body::cuboid([4.0, 4.0, 4.0])
            .translated([2.0, 2.0, 2.0])
            .unwrap();
        check(&a, &b, BooleanOp::Intersection, 8.0);
        check(&a, &b, BooleanOp::Difference, 56.0);
        check(&a, &b, BooleanOp::Union, 120.0);
    }

    #[test]
    fn a_corner_cut_has_occts_topology() {
        // Sides chained between the same two faces: a corner cut out of a
        // box is 9 faces, 21 edges and 14 vertices, as OpenCASCADE has it.
        let a = Body::cuboid([4.0, 4.0, 4.0]);
        let b = Body::cuboid([4.0, 4.0, 4.0])
            .translated([2.0, 2.0, 2.0])
            .unwrap();
        let r = planar(&a, &b, BooleanOp::Difference, 1e-6)
            .unwrap()
            .unwrap();
        assert_eq!(
            (r.faces().len(), r.edges().len(), r.vertices().len()),
            (9, 21, 14)
        );
    }

    #[test]
    fn boxes_sharing_part_of_a_wall() {
        // Side by side, offset: the walls between them overlap in part, and
        // the tops and bottoms lie in one plane.
        let a = Body::cuboid([2.0, 2.0, 2.0]);
        let b = Body::cuboid([2.0, 2.0, 2.0])
            .translated([2.0, 1.0, 0.0])
            .unwrap();
        check(&a, &b, BooleanOp::Union, 16.0);
        check(&a, &b, BooleanOp::Intersection, 0.0);
        check(&a, &b, BooleanOp::Difference, 8.0);
    }

    #[test]
    fn a_slot_cut_through_to_the_floor() {
        // A tool whose bottom lies on the floor and whose ends stand out of
        // both sides: the bar comes back as two blocks.
        let a = Body::cuboid([6.0, 2.0, 2.0]);
        let b = Body::cuboid([2.0, 4.0, 2.0])
            .translated([0.0, 0.0, 1.0])
            .unwrap();
        check(&a, &b, BooleanOp::Difference, 24.0 - 4.0);
        check(&a, &b, BooleanOp::Union, 24.0 + 16.0 - 4.0);
        check(&a, &b, BooleanOp::Intersection, 4.0);
    }

    #[test]
    fn a_box_inside_another_leaves_a_cavity() {
        let a = Body::cuboid([4.0, 4.0, 4.0]);
        let b = Body::cuboid([2.0, 2.0, 2.0]);
        check(&a, &b, BooleanOp::Difference, 56.0);
        check(&a, &b, BooleanOp::Intersection, 8.0);
        check(&a, &b, BooleanOp::Union, 64.0);
    }

    #[test]
    fn a_bar_cut_by_a_cylinder_through_its_floor() {
        let a = Body::cuboid([4.0, 4.0, 2.0]);
        // Through the floor and out of one side, clear of the top.
        let b = Body::cylinder([0.0, 0.5, -0.5], [0.0, 1.0, 0.0], 1.2, 3.0);
        for op in [
            BooleanOp::Difference,
            BooleanOp::Union,
            BooleanOp::Intersection,
        ] {
            let r = general(&a, &b, op, 1e-3).unwrap().unwrap();
            assert!(r.is_valid_solid(1e-3), "{op:?}: {:?}", r.defects(1e-3));
        }
    }

    #[test]
    fn a_collar_on_a_plate_round_its_hole() {
        // The hole's rim and the collar's inner rim are one circle sampled
        // two ways, and both the plate's top and the collar's foot lie in
        // one plane.
        let plate = Body::cuboid([40.0, 40.0, 10.0])
            .boolean(
                &Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 2.5, 12.0),
                BooleanOp::Difference,
                1e-3,
            )
            .unwrap();
        let collar = Body::cylinder([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 4.5, 1.2)
            .boolean(
                &Body::cylinder([0.0, 0.0, 4.0], [0.0, 0.0, 1.0], 2.5, 3.2),
                BooleanOp::Difference,
                1e-3,
            )
            .unwrap();
        let r = general(&plate, &collar, BooleanOp::Union, 1e-3)
            .unwrap()
            .unwrap();
        assert!(r.is_valid_solid(1e-3), "{:?}", r.defects(1e-3));
    }

    #[test]
    fn spheres_that_kiss_graze() {
        let a = Body::sphere([0.0; 3], 1.0);
        let b = Body::sphere([2.0, 0.0, 0.0], 1.0);
        for op in [
            BooleanOp::Difference,
            BooleanOp::Union,
            BooleanOp::Intersection,
        ] {
            assert!(matches!(
                general(&a, &b, op, 1e-3),
                Some(Err(Declined::TangentialContact { .. }))
            ));
        }
    }

    /// The volume of a body bounded by planes and cylinders, from its faces:
    /// `(p·n)/3` over each, a plane by its loops' area, a cylinder by Green's
    /// theorem in its parameters, where `p·n` is `r + c·n(u)` and `dA` is
    /// `r du dv`. Not from a tessellation, which has errors of its own.
    fn exact_volume(r: &Body) -> f64 {
        r.faces()
            .iter()
            .map(|f| {
                let sign = if f.flipped { -1.0 } else { 1.0 };
                let loops = f.loops.as_ref().expect("stated loops");
                sign * match &r.surfaces()[f.surface] {
                    Surface::Plane { origin, normal, .. } => {
                        v3::dot(*origin, *normal) * loops.iter().map(|l| l.area).sum::<f64>() / 3.0
                    }
                    Surface::Cylinder {
                        origin,
                        axis,
                        x_dir,
                        radius,
                    } => {
                        let y = v3::cross(*axis, *x_dir);
                        let (cx, cy) = (v3::dot(*origin, *x_dir), v3::dot(*origin, y));
                        let big =
                            |u: f64| radius * (radius * u + cx * u.sin() - cy * u.cos()) / 3.0;
                        let mut sum = 0.0;
                        for l in loops {
                            let n = l.uv.len();
                            for k in 0..n {
                                let (a, b) = (l.uv[k], l.uv[(k + 1) % n]);
                                let m = (a[0] + b[0]) / 2.0;
                                sum += (big(a[0]) + 4.0 * big(m) + big(b[0])) / 6.0 * (b[1] - a[1]);
                            }
                        }
                        sum
                    }
                    other => panic!("no exact volume for a {}", other.kind()),
                }
            })
            .sum()
    }

    #[test]
    fn a_bore_through_a_pockets_end_wall() {
        // PartDesignExample's second pocket: a cylinder whose axis lies in
        // the end wall of an earlier pocket. The volume it takes is the
        // disc's area over its height, less the part of the disc over the
        // pocket — ∫ √(r² − y²) dy from −min(r, 25) to −5 — over the
        // pocket's depth of 15. At r = 25 the cylinder touches the pocket's
        // side wall along its corner; at 24.7 it clears it. Right, or
        // declined. (Its mesh was 86 out here: the fill of a notched band,
        // not the solid.)
        let block = Body::cuboid([140.0, 70.0, 100.0]);
        let pocket = Body::cuboid([100.0, 20.0, 15.0])
            .translated([-12.5, -15.0, 42.5])
            .unwrap();
        let base = general(&block, &pocket, BooleanOp::Difference, 1e-4)
            .unwrap()
            .unwrap();
        for r in [25.0f64, 24.7] {
            let bore = Body::cylinder([37.5, 0.0, 0.0], [0.0, 0.0, 1.0], r, 50.0);
            let f = |y: f64| {
                y / 2.0 * (r * r - y * y).max(0.0).sqrt()
                    + r * r / 2.0 * (y / r).clamp(-1.0, 1.0).asin()
            };
            let strip = f(-5.0) - f(-r.min(25.0));
            let removed = std::f64::consts::PI * r * r * 50.0 - strip * 15.0;
            let expected = 140.0 * 70.0 * 100.0 - 100.0 * 20.0 * 15.0 - removed;
            if let Ok(out) = general(&base, &bore, BooleanOp::Difference, 1e-4).unwrap() {
                assert!(out.is_valid_solid(1e-4), "{:?}", out.defects(1e-4));
                let got = exact_volume(&out);
                assert!(
                    (got - expected).abs() < 0.1,
                    "r {r}: volume {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn a_plate_drilled_through() {
        let plate = Body::cuboid([10.0, 8.0, 2.0]);
        let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0);
        let r = general(&plate, &drill, BooleanOp::Difference, 1e-3)
            .unwrap()
            .unwrap();
        assert!(r.is_valid_solid(1e-3), "{:?}", r.defects(1e-3));
        // Top, bottom, four sides, and the bore as one face with its seam.
        assert_eq!(r.faces().len(), 7);
    }
}
