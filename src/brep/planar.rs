//! Subdividing a face's parameter region by the curves drawn on it.
//!
//! This is the piece every remaining `Declined::NeedsArrangement` was waiting
//! for. Splitting a face is easy when an intersection curve closes on it — the
//! curve is a hole and the region around it is the rest — and that case is
//! handled directly. Everything else needs a real subdivision: a curve that
//! *enters and leaves* through the face's boundary cuts it in two, several such
//! curves cut it into several pieces, and which piece is which is not something
//! a containment test can answer.
//!
//! Rounding an edge, cutting a slot, clipping a corner: all of them are a face
//! crossed by open curves, and none of them is exotic.
//!
//! The approach is the standard one — build a planar graph of the boundary plus
//! the chords, then walk its half-edges to read the faces off. Kept in its own
//! module and in plain 2D coordinates, because it is entirely a question about
//! polygons and is worth testing as one, without constructing a solid to ask.

/// Endpoints closer than this fraction of the region's extent are the same
/// point. Relative, because parameter spaces are not all the same size: a
/// plane's is in model units, a cylinder's `u` is radians.
const SNAP: f64 = 1e-7;

/// A closed ring of parameter-space points, and where each came from.
///
/// `sources` carries, per point, the index of the input ring or chord it was
/// taken from — the caller needs it to get back to vertex indices, which is
/// what keeps a seam shared rather than merely coincident.
#[derive(Debug, Clone, PartialEq)]
pub struct Ring {
    pub uv: Vec<[f64; 2]>,
    pub sources: Vec<Source>,
}

/// Where a subdivided point came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Point `i` of the outer boundary.
    Boundary(usize),
    /// Point `i` of chord `c`.
    Chord { chord: usize, point: usize },
    /// A crossing computed here, belonging to no input.
    Crossing,
}

/// One piece of a subdivided face.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub outer: Ring,
    /// Rings enclosed by `outer` that are *not* part of it.
    ///
    /// A cut does not always separate a region in two: a closed curve inside one
    /// leaves it connected, with a hole. The walk finds such a ring wound the
    /// other way, and it belongs to whichever region encloses it.
    pub holes: Vec<Ring>,
    /// Signed area of `outer`, positive.
    pub area: f64,
}

/// Cut `boundary` with `chords`, returning the pieces.
///
/// Each chord is a polyline whose two ends lie *on* the boundary; a chord that
/// stops inside is not a cut and the whole thing is declined with `None`,
/// because half a cut is not a subdivision and guessing the rest is exactly what
/// this layer does not do.
///
/// Chords that cross each other are handled: the crossings are computed and
/// become nodes, so two cuts across a face give four pieces rather than a
/// refusal. That case is not exotic either — it is what two boxes overlapping at
/// a corner puts on every face between them.
/// Clip a path to the inside of `boundary`, returning the stretches that lie
/// within it.
///
/// This is what makes a *boolean* out of a subdivision. Cutting a region by
/// another region's outline only works when that outline is already inside;
/// when the two overlap partly, the outline has to be trimmed to the part that
/// is, and each surviving stretch then enters and leaves through the boundary —
/// which is exactly a chord, and the rest of this module already handles those.
///
/// A path entirely inside comes back unchanged; one entirely outside comes back
/// empty.
pub fn clip_to_region(boundary: &[[f64; 2]], path: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
    if boundary.len() < 3 || path.len() < 2 {
        return Vec::new();
    }
    let closed = dist(path[0], *path.last().unwrap()) <= extent(boundary) * SNAP;

    // Every point of the path, with the boundary crossings spliced in.
    let mut pts: Vec<([f64; 2], bool)> = Vec::new(); // (point, is a crossing)
    for i in 0..path.len() - 1 {
        pts.push((path[i], false));
        let mut here: Vec<(f64, [f64; 2])> = Vec::new();
        for j in 0..boundary.len() {
            let (q, q2) = (boundary[j], boundary[(j + 1) % boundary.len()]);
            if let Some((t, _, at)) = segment_cross(path[i], path[i + 1], q, q2) {
                if !here.iter().any(|(_, p)| dist(*p, at) <= SNAP) {
                    here.push((t, at));
                }
            }
        }
        here.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pts.extend(here.into_iter().map(|(_, p)| (p, true)));
    }
    pts.push((*path.last().unwrap(), false));

    // Runs between crossings, kept when they are inside.
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut run: Vec<[f64; 2]> = Vec::new();
    for (i, (p, crossing)) in pts.iter().enumerate() {
        run.push(*p);
        if *crossing && i + 1 < pts.len() {
            if run.len() > 1 && inside_run(boundary, &run) {
                out.push(std::mem::take(&mut run));
                run.push(*p);
            } else {
                run.clear();
                run.push(*p);
            }
        }
    }
    if run.len() > 1 && inside_run(boundary, &run) {
        out.push(run);
    }

    // A closed path's first and last runs are one run through the seam.
    if closed && out.len() > 1 {
        let first = out[0].clone();
        let last = out[out.len() - 1].clone();
        if dist(*last.last().unwrap(), first[0]) <= extent(boundary) * SNAP {
            let mut joined = last;
            joined.extend(first.into_iter().skip(1));
            let n = out.len();
            out[0] = joined;
            out.remove(n - 1);
        }
    }
    out
}

/// Is the middle of this run inside the boundary?
fn inside_run(boundary: &[[f64; 2]], run: &[[f64; 2]]) -> bool {
    let mid = run[run.len() / 2];
    let probe = if run.len() >= 2 {
        let a = run[run.len() / 2 - 1];
        [(a[0] + mid[0]) * 0.5, (a[1] + mid[1]) * 0.5]
    } else {
        mid
    };
    point_in_ring(boundary, probe)
}

/// Where two segments cross, as `(parameter on a, parameter on b, point)`.
fn segment_cross(
    p: [f64; 2],
    p2: [f64; 2],
    q: [f64; 2],
    q2: [f64; 2],
) -> Option<(f64, f64, [f64; 2])> {
    let r = [p2[0] - p[0], p2[1] - p[1]];
    let s = [q2[0] - q[0], q2[1] - q[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() <= f64::MIN_POSITIVE {
        return None;
    }
    let d = [q[0] - p[0], q[1] - p[1]];
    let t = (d[0] * s[1] - d[1] * s[0]) / denom;
    let u = (d[0] * r[1] - d[1] * r[0]) / denom;
    if !(-SNAP..=1.0 + SNAP).contains(&t) || !(-SNAP..=1.0 + SNAP).contains(&u) {
        return None;
    }
    Some((t, u, [p[0] + r[0] * t, p[1] + r[1] * t]))
}

pub fn subdivide(boundary: &[[f64; 2]], chords: &[Vec<[f64; 2]>]) -> Option<Vec<Region>> {
    if boundary.len() < 3 {
        return None;
    }
    if chords.is_empty() {
        let ring = Ring {
            uv: boundary.to_vec(),
            sources: (0..boundary.len()).map(Source::Boundary).collect(),
        };
        let area = signed_area(&ring.uv).abs();
        return Some(vec![Region {
            outer: ring,
            holes: Vec::new(),
            area,
        }]);
    }

    // 1. Where chords meet each other. Computed first, because a chord end is
    //    allowed to land on another chord rather than on the boundary — and it
    //    usually does. Two boxes overlapping at a corner put two cuts on the
    //    face between them that *stop at the corner they share*: neither
    //    reaches the boundary at both ends, and together they cross the face.
    let mut nodes: Vec<Node> = boundary
        .iter()
        .enumerate()
        .map(|(i, p)| Node {
            uv: *p,
            source: Source::Boundary(i),
        })
        .collect();
    let scale = extent(boundary).max(1e-12);
    let snap = scale * SNAP;

    let mut cuts: Vec<Vec<(f64, usize)>> = vec![Vec::new(); chords.len()];
    for i in 0..chords.len() {
        for j in i + 1..chords.len() {
            for (ti, tj, at) in crossings(&chords[i], &chords[j]) {
                let node = match nodes.iter().position(|n| dist(n.uv, at) <= snap) {
                    Some(n) => n,
                    None => {
                        nodes.push(Node {
                            uv: at,
                            source: Source::Crossing,
                        });
                        nodes.len() - 1
                    }
                };
                cuts[i].push((ti, node));
                cuts[j].push((tj, node));
            }
        }
    }

    // 2. Every chord end has to land on the boundary or on such a junction —
    //    otherwise it stops in open space, which is half a cut. Where it lands
    //    on the boundary, the boundary needs a vertex there, or the two pieces
    //    meet along an edge one of them has split and the other has not.
    let mut splits: Vec<Vec<(f64, usize)>> = vec![Vec::new(); boundary.len()];
    let mut ends: Vec<[usize; 2]> = Vec::with_capacity(chords.len());

    for (ci, chord) in chords.iter().enumerate() {
        if chord.len() < 2 {
            return None;
        }
        // A path that comes back to where it started is not a cut across the
        // region — it is a ring *inside* it, and the region stays connected with
        // a hole. It lands on nothing and needs a node of its own.
        if chord.len() > 3 && dist(chord[0], *chord.last().unwrap()) <= snap {
            nodes.push(Node {
                uv: chord[0],
                source: Source::Chord {
                    chord: ci,
                    point: 0,
                },
            });
            let n = nodes.len() - 1;
            ends.push([n, n]);
            continue;
        }
        let mut got = [usize::MAX; 2];
        for (k, p, t_end) in [
            (0usize, chord[0], 0.0),
            (1, *chord.last().unwrap(), (chord.len() - 1) as f64),
        ] {
            // Already a junction with another chord?
            if let Some((_, n)) = cuts[ci].iter().find(|(t, _)| (t - t_end).abs() <= SNAP) {
                got[k] = *n;
                continue;
            }
            let (seg, t, at) = nearest_on_boundary(boundary, p)?;
            if dist(at, p) > snap {
                return None; // stops in open space: not a cut
            }
            // Reuse an existing corner when the end lands on one.
            got[k] = if t <= SNAP {
                seg
            } else if t >= 1.0 - SNAP {
                (seg + 1) % boundary.len()
            } else {
                match splits[seg].iter().find(|(u, _)| (u - t).abs() <= SNAP) {
                    Some((_, n)) => *n,
                    None => {
                        // The chord's *own* endpoint, not an anonymous crossing.
                        //
                        // It is a point of the curve, and the face on the other
                        // side of that curve names it by index. Calling it a
                        // crossing loses that: the caller has nothing to map it
                        // back to and welds it by position instead, which lands
                        // a hair off and leaves the two faces joining the same
                        // place through different vertices.
                        // An anonymous crossing, welded by position later.
                        //
                        // Naming it by the chord's own vertex would be better —
                        // the face on the other side of that curve names it by
                        // index — but the position and the index then disagree
                        // by up to `snap`, whichever of the two is used for the
                        // node, and the weld puts them in different places. That
                        // costs more than it buys until the chord's ends are
                        // made to land *exactly* on the boundary rather than
                        // within tolerance of it.
                        nodes.push(Node {
                            uv: at,
                            source: Source::Crossing,
                        });
                        splits[seg].push((t, nodes.len() - 1));
                        nodes.len() - 1
                    }
                }
            };
        }
        if got[0] == got[1] {
            return None; // a chord that starts and ends at one point
        }
        ends.push(got);
    }

    // 3. The graph: boundary segments in order, each subdivided by whatever
    //    landed on it, then each chord as one edge per stretch between nodes.
    let mut edges: Vec<Edge> = Vec::new();
    for (i, here) in splits.iter().enumerate() {
        let mut here = here.clone();
        here.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut from = i;
        for (_, n) in here {
            edges.push(Edge {
                from,
                to: n,
                interior: Vec::new(),
                chord: None,
            });
            from = n;
        }
        edges.push(Edge {
            from,
            to: (i + 1) % boundary.len(),
            interior: Vec::new(),
            chord: None,
        });
    }
    for (ci, chord) in chords.iter().enumerate() {
        let mut stops: Vec<(f64, usize)> = cuts[ci].clone();
        stops.push((0.0, ends[ci][0]));
        stops.push(((chord.len() - 1) as f64, ends[ci][1]));
        stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        stops.dedup_by(|a, b| (a.0 - b.0).abs() <= SNAP);
        // A closed ring is one edge from its node back to itself.
        if ends[ci][0] == ends[ci][1] && cuts[ci].is_empty() {
            edges.push(Edge {
                from: ends[ci][0],
                to: ends[ci][0],
                interior: chord[1..chord.len() - 1].to_vec(),
                chord: Some(ci),
            });
            continue;
        }
        for w in stops.windows(2) {
            let (t0, n0) = w[0];
            let (t1, n1) = w[1];
            if n0 == n1 {
                continue;
            }
            edges.push(Edge {
                from: n0,
                to: n1,
                interior: between(chord, t0, t1),
                chord: Some(ci),
            });
        }
    }

    // 4. Walk the half-edges. At each node take the next one clockwise from the
    //    reverse of the one just travelled, which traces every face exactly once.
    let half: Vec<Half> = edges
        .iter()
        .enumerate()
        .flat_map(|(i, e)| {
            [
                Half {
                    edge: i,
                    from: e.from,
                    to: e.to,
                    forward: true,
                },
                Half {
                    edge: i,
                    from: e.to,
                    to: e.from,
                    forward: false,
                },
            ]
        })
        .collect();

    let mut out_of: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, h) in half.iter().enumerate() {
        out_of[h.from].push(i);
    }
    // Sort each node's outgoing half-edges by direction.
    for (n, list) in out_of.iter_mut().enumerate() {
        list.sort_by(|&a, &b| {
            angle_of(&half[a], &nodes, &edges, n)
                .partial_cmp(&angle_of(&half[b], &nodes, &edges, n))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut seen = vec![false; half.len()];
    let mut regions: Vec<Region> = Vec::new();
    let mut inverted: Vec<(Ring, f64)> = Vec::new();
    for start in 0..half.len() {
        if seen[start] {
            continue;
        }
        // Every half-edge contributes its own `from`, so a node the walk passes
        // through twice is emitted twice — and it must be. A chord that dangles
        // is traversed out and back, and the ring leaves and re-enters the
        // boundary at the same node; written once, the ring instead jumps from
        // the chord's far end to whatever came after that node, and the wedge
        // between the two legs is bounded by nothing.
        //
        // That is what `ball - bore - cross` stores at 2e-4: `… 688, 371 …`
        // where the walk produced `… 688, 370, 371 …`, and the fill leaves the
        // two legs open. The loss is downstream of here — this walk is right —
        // so whatever copies a region into a `TrimLoop` is dropping one of the
        // two, and that is the next thing to look at.
        let mut ring: Vec<[f64; 2]> = Vec::new();
        let mut sources: Vec<Source> = Vec::new();
        let mut h = start;
        loop {
            if seen[h] {
                break;
            }
            seen[h] = true;
            let cur = &half[h];
            ring.push(nodes[cur.from].uv);
            sources.push(nodes[cur.from].source);
            let e = &edges[cur.edge];
            if !e.interior.is_empty() {
                let chord = e.chord.expect("only chords carry interior points");
                let n = e.interior.len();
                for k in 0..n {
                    let k = if cur.forward { k } else { n - 1 - k };
                    ring.push(e.interior[k]);
                    sources.push(Source::Chord {
                        chord,
                        point: k + 1,
                    });
                }
            }
            // Turn as tightly as possible: the reverse half-edge, then the next
            // one round in order.
            let back = h ^ 1;
            let at = half[back].from;
            let list = &out_of[at];
            let pos = list.iter().position(|&x| x == back)?;
            h = list[(pos + list.len() - 1) % list.len()];
            if h == start {
                break;
            }
        }
        if ring.len() < 3 {
            continue;
        }
        let area = signed_area(&ring);
        if area > 0.0 {
            regions.push(Region {
                outer: Ring { uv: ring, sources },
                holes: Vec::new(),
                area,
            });
        } else {
            // Wound the other way: either the unbounded face, or a ring sitting
            // inside a region as a hole. Which it is depends on whether anything
            // encloses it.
            inverted.push((Ring { uv: ring, sources }, -area));
        }
    }

    if regions.is_empty() {
        return None;
    }

    // Give each inverted ring to the smallest region enclosing it. The one
    // nothing encloses is the unbounded face and is dropped.
    for (ring, area) in inverted {
        let Some(probe) = ring.uv.first().copied() else {
            continue;
        };
        let mut best: Option<usize> = None;
        for (i, r) in regions.iter().enumerate() {
            // Strictly larger, with a margin. A ring and its own reversal have
            // the same area to the last bit, and the probe — a vertex of the
            // ring — lies exactly on that reversal's boundary, where
            // `point_in_ring` may answer either way. Rounding an edge is that
            // case: four chords close a cycle inside the face, and the cycle
            // was handed to itself as a hole. Its area then cancelled to
            // nothing, and the region that really contained it never got it.
            if r.area <= area * (1.0 + 1e-9) || !point_in_ring(&r.outer.uv, probe) {
                continue;
            }
            if best.is_none_or(|b| r.area < regions[b].area) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            regions[i].holes.push(ring);
            regions[i].area -= area;
        }
    }
    Some(regions)
}

struct Node {
    uv: [f64; 2],
    source: Source,
}

struct Edge {
    from: usize,
    to: usize,
    /// Points strictly between the ends, in `from → to` order.
    interior: Vec<[f64; 2]>,
    chord: Option<usize>,
}

struct Half {
    edge: usize,
    from: usize,
    to: usize,
    forward: bool,
}

/// The direction a half-edge leaves its node, as an angle.
///
/// Taken from the first *interior* point when there is one, not from the far
/// end: a chord that bows away and comes back would otherwise be sorted by
/// where it ends rather than by where it goes, and the walk would turn the
/// wrong way.
fn angle_of(h: &Half, nodes: &[Node], edges: &[Edge], at: usize) -> f64 {
    let e = &edges[h.edge];
    let next = if e.interior.is_empty() {
        nodes[h.to].uv
    } else if h.forward {
        e.interior[0]
    } else {
        e.interior[e.interior.len() - 1]
    };
    let from = nodes[at].uv;
    (next[1] - from[1]).atan2(next[0] - from[0])
}

/// Winding-number containment, matching the caller's own test.
fn point_in_ring(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = ring.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.0
}

/// The polyline points strictly between two parameters, where a parameter is
/// `segment index + fraction along it`.
fn between(chord: &[[f64; 2]], t0: f64, t1: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let first = (t0.floor() as usize) + 1;
    for (k, p) in chord.iter().enumerate().skip(first) {
        if (k as f64) >= t1 - SNAP {
            break;
        }
        out.push(*p);
    }
    out
}

/// Where two polylines cross, as `(parameter on a, parameter on b, point)`.
///
/// Segment ends are *included*, because a crossing can land exactly on a
/// polyline vertex — a sampled curve crossing a cut through its midpoint does
/// it every time — and rejecting endpoints makes both adjacent segments
/// disown it. Such a crossing is then found twice, so the results are deduped
/// by position.
fn crossings(a: &[[f64; 2]], b: &[[f64; 2]]) -> Vec<(f64, f64, [f64; 2])> {
    let mut out = Vec::new();
    for i in 0..a.len().saturating_sub(1) {
        for j in 0..b.len().saturating_sub(1) {
            let (p, p2) = (a[i], a[i + 1]);
            let (q, q2) = (b[j], b[j + 1]);
            let r = [p2[0] - p[0], p2[1] - p[1]];
            let s = [q2[0] - q[0], q2[1] - q[1]];
            let denom = r[0] * s[1] - r[1] * s[0];
            if denom.abs() <= f64::MIN_POSITIVE {
                continue; // parallel or degenerate
            }
            let d = [q[0] - p[0], q[1] - p[1]];
            let t = (d[0] * s[1] - d[1] * s[0]) / denom;
            let u = (d[0] * r[1] - d[1] * r[0]) / denom;
            if !(-SNAP..=1.0 + SNAP).contains(&t) || !(-SNAP..=1.0 + SNAP).contains(&u) {
                continue;
            }
            let at = [p[0] + r[0] * t, p[1] + r[1] * t];
            if out
                .iter()
                .any(|&(_, _, q): &(f64, f64, [f64; 2])| dist(q, at) <= SNAP)
            {
                continue;
            }
            out.push((i as f64 + t, j as f64 + u, at));
        }
    }
    out
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn extent(ring: &[[f64; 2]]) -> f64 {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in ring {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (hi[0] - lo[0]).max(hi[1] - lo[1])
}

/// The closest point of the boundary to `p`, as `(segment, parameter, point)`.
fn nearest_on_boundary(boundary: &[[f64; 2]], p: [f64; 2]) -> Option<(usize, f64, [f64; 2])> {
    let n = boundary.len();
    let mut best: Option<(f64, usize, f64, [f64; 2])> = None;
    for i in 0..n {
        let (a, b) = (boundary[i], boundary[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 <= f64::MIN_POSITIVE {
            0.0
        } else {
            (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        };
        let at = [a[0] + dx * t, a[1] + dy * t];
        let d = dist(at, p);
        if best.is_none_or(|(bd, _, _, _)| d < bd) {
            best = Some((d, i, t, at));
        }
    }
    best.map(|(_, i, t, at)| (i, t, at))
}


/// Split every segment where it crosses another, so what comes out is a planar
/// graph rather than a set of loops that may or may not be simple.
///
/// This is the first half of filling a face without trusting its rings. A trim
/// loop can cross itself — 17 of 49 do, in this crate's own results, because a
/// traced curve's closure keeps a sample that has already gone round — and no
/// triangulation of a self-crossing polygon is right. Every repair at the ring
/// has failed, five of them, because a ring is a curve *projected*, and editing
/// one projection breaks its agreement with the face across the boundary.
///
/// An arrangement does not care. A crossing becomes a vertex, and what was an
/// impossible polygon becomes an ordinary graph whose faces can be walked.
///
/// Points are welded at `tolerance`, so a crossing that lands on an existing
/// vertex is that vertex rather than a second one a whisker away.
pub fn arrange(rings: &[Vec<[f64; 2]>], tolerance: f64) -> (Vec<[f64; 2]>, Vec<[usize; 2]>) {
    // Every ring's segments, as endpoints.
    let mut ends: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for r in rings {
        let n = r.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            ends.push((r[i], r[(i + 1) % n]));
        }
    }

    // Where each segment is cut, as fractions along it.
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); ends.len()];
    for i in 0..ends.len() {
        for j in i + 1..ends.len() {
            let (a, b) = ends[i];
            let (c, d) = ends[j];
            let r = [b[0] - a[0], b[1] - a[1]];
            let s = [d[0] - c[0], d[1] - c[1]];
            let denom = r[0] * s[1] - r[1] * s[0];
            if denom.abs() < 1e-15 {
                continue; // parallel, or both degenerate
            }
            let ac = [c[0] - a[0], c[1] - a[1]];
            let t = (ac[0] * s[1] - ac[1] * s[0]) / denom;
            let u = (ac[0] * r[1] - ac[1] * r[0]) / denom;
            // Strictly interior to both: an endpoint touching is already a
            // vertex, and cutting there would only duplicate it.
            if t > 1e-12 && t < 1.0 - 1e-12 && u > 1e-12 && u < 1.0 - 1e-12 {
                cuts[i].push(t);
                cuts[j].push(u);
            }
        }
    }

    // Weld as we go: a crossing computed from two directions lands twice.
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let id = |p: [f64; 2], pts: &mut Vec<[f64; 2]>| -> usize {
        for (i, q) in pts.iter().enumerate() {
            if (q[0] - p[0]).hypot(q[1] - p[1]) <= tolerance {
                return i;
            }
        }
        pts.push(p);
        pts.len() - 1
    };

    let mut edges: Vec<[usize; 2]> = Vec::new();
    for (i, (a, b)) in ends.iter().enumerate() {
        let mut ts = cuts[i].clone();
        ts.push(0.0);
        ts.push(1.0);
        ts.sort_by(|x, y| x.total_cmp(y));
        ts.dedup_by(|x, y| (*x - *y).abs() <= 1e-12);
        for w in ts.windows(2) {
            let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let (p, q) = (id(at(w[0]), &mut pts), id(at(w[1]), &mut pts));
            if p != q {
                edges.push([p, q]);
            }
        }
    }
    (pts, edges)
}

/// The cycles of the arrangement that lie *inside* the rings, by winding.
///
/// [`arrange`] turns crossing loops into a graph; this walks that graph's faces
/// and keeps the ones the rings enclose. Winding is what makes it independent of
/// the rings' own orientation — a loop that laps itself has no consistent
/// orientation to trust, which is exactly why every repair that trusted one has
/// failed.
///
/// Each cycle comes back wound counter-clockwise, so a caller can tell an outer
/// boundary from a hole by the sign it would have had.
pub fn regions_of(rings: &[Vec<[f64; 2]>], tolerance: f64) -> Vec<Vec<[f64; 2]>> {
    let (pts, edges) = arrange(rings, tolerance);
    if pts.len() < 3 || edges.is_empty() {
        return Vec::new();
    }

    // Half-edges, and the ones leaving each vertex sorted by direction.
    let mut half: Vec<(usize, usize)> = Vec::with_capacity(edges.len() * 2);
    for e in &edges {
        half.push((e[0], e[1]));
        half.push((e[1], e[0]));
    }
    let angle = |h: usize| -> f64 {
        let (a, b) = half[h];
        (pts[b][1] - pts[a][1]).atan2(pts[b][0] - pts[a][0])
    };
    let mut out_of: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for h in 0..half.len() {
        out_of[half[h].0].push(h);
    }
    for list in out_of.iter_mut() {
        list.sort_by(|&x, &y| angle(x).total_cmp(&angle(y)));
    }

    // Walk each face by taking the tightest turn: the reverse half-edge's
    // predecessor in angular order.
    let mut seen = vec![false; half.len()];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..half.len() {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut h = start;
        loop {
            if seen[h] {
                break;
            }
            seen[h] = true;
            cycle.push(half[h].0);
            let back = h ^ 1;
            let at = half[back].0;
            let list = &out_of[at];
            let Some(pos) = list.iter().position(|&x| x == back) else {
                break;
            };
            h = list[(pos + list.len() - 1) % list.len()];
            if h == start {
                break;
            }
        }
        if cycle.len() >= 3 {
            cycles.push(cycle);
        }
    }

    // Keep the cycles the rings enclose. The outer face of an arrangement winds
    // the other way, and a lap encloses nothing.
    let mut kept = Vec::new();
    for cycle in cycles {
        let uv: Vec<[f64; 2]> = cycle.iter().map(|&i| pts[i]).collect();
        let area = signed_area(&uv);
        if area <= tolerance * tolerance {
            continue;
        }
        let inside = uv
            .iter()
            .zip(uv.iter().cycle().skip(1))
            .fold([0.0, 0.0], |acc, (a, b)| {
                [acc[0] + (a[0] + b[0]) * 0.5, acc[1] + (a[1] + b[1]) * 0.5]
            });
        let probe = [inside[0] / uv.len() as f64, inside[1] / uv.len() as f64];
        if winding(rings, probe) != 0 {
            kept.push(uv);
        }
    }
    kept
}

/// How many times the rings wind around `p`, counted by crossings of a ray.
fn winding(rings: &[Vec<[f64; 2]>], p: [f64; 2]) -> i32 {
    let mut w = 0;
    for r in rings {
        let n = r.len();
        for i in 0..n {
            let (a, b) = (r[i], r[(i + 1) % n]);
            if a[1] <= p[1] {
                if b[1] > p[1]
                    && (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]) > 0.0
                {
                    w += 1;
                }
            } else if b[1] <= p[1]
                && (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]) < 0.0
            {
                w -= 1;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    }

    /// Every piece's area, sorted — the check that matters, since the pieces
    /// must tile the original exactly.
    fn areas(regions: &[Region]) -> Vec<f64> {
        let mut a: Vec<f64> = regions
            .iter()
            .map(|r| (r.area * 1e6).round() / 1e6)
            .collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        a
    }

    #[test]
    fn no_chords_leaves_the_region_alone() {
        let r = subdivide(&square(), &[]).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].area, 16.0);
    }

    #[test]
    fn one_chord_cuts_the_region_in_two() {
        // Straight across the middle: two 4 × 2 halves.
        let r = subdivide(&square(), &[vec![[0.0, 2.0], [4.0, 2.0]]]).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(areas(&r), vec![8.0, 8.0]);
    }

    #[test]
    fn a_chord_landing_on_corners_reuses_them() {
        // A diagonal: two triangles, and no new boundary vertices.
        let r = subdivide(&square(), &[vec![[0.0, 0.0], [4.0, 4.0]]]).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(areas(&r), vec![8.0, 8.0]);
    }

    #[test]
    fn two_chords_cut_three_pieces() {
        // Two parallel cuts — a slot's seams on a face.
        let r = subdivide(
            &square(),
            &[vec![[0.0, 1.0], [4.0, 1.0]], vec![[0.0, 3.0], [4.0, 3.0]]],
        )
        .unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(areas(&r), vec![4.0, 8.0, 4.0].tap_sort());
    }

    #[test]
    fn a_curved_chord_keeps_its_shape() {
        // A chord is a polyline, not a segment: the points between its ends are
        // the sampled curve and have to survive into the pieces, or a rounded
        // edge comes out chamfered.
        let bow: Vec<[f64; 2]> = (0..=8)
            .map(|i| {
                let t = i as f64 / 8.0;
                [4.0 * t, 2.0 + (std::f64::consts::PI * t).sin()]
            })
            .collect();
        let r = subdivide(&square(), std::slice::from_ref(&bow)).unwrap();
        assert_eq!(r.len(), 2);
        // The bow bulges upward, so the lower piece is the larger one.
        let total: f64 = r.iter().map(|x| x.area).sum();
        assert!((total - 16.0).abs() < 1e-9, "pieces total {total}");
        assert!(
            r.iter().any(|x| x.area > 8.5),
            "the bow should make the halves unequal: {:?}",
            areas(&r)
        );
        // Every interior point of the chord appears in some piece.
        for p in &bow[1..bow.len() - 1] {
            assert!(
                r.iter()
                    .any(|x| x.outer.uv.iter().any(|q| dist(*q, *p) < 1e-9)),
                "{p:?} was dropped"
            );
        }
    }

    #[test]
    fn the_pieces_tile_the_original() {
        // Whatever the cuts, the areas must add back up.
        for chords in [
            vec![vec![[0.0, 2.0], [4.0, 2.0]]],
            vec![vec![[2.0, 0.0], [2.0, 4.0]]],
            vec![vec![[0.0, 1.0], [4.0, 3.0]]],
            vec![
                vec![[0.0, 1.0], [4.0, 1.0]],
                vec![[0.0, 2.0], [4.0, 2.0]],
                vec![[0.0, 3.0], [4.0, 3.0]],
            ],
            vec![vec![[0.0, 0.0], [4.0, 2.0]], vec![[0.0, 4.0], [4.0, 2.0]]],
        ] {
            let r = subdivide(&square(), &chords).unwrap();
            let total: f64 = r.iter().map(|x| x.area).sum();
            assert!(
                (total - 16.0).abs() < 1e-9,
                "{chords:?} gave {} pieces totalling {total}",
                r.len()
            );
        }
    }

    #[test]
    fn a_chord_that_stops_inside_is_refused() {
        // Half a cut is not a subdivision, and finishing it by guessing is
        // exactly what this layer does not do.
        assert!(subdivide(&square(), &[vec![[0.0, 2.0], [2.0, 2.0]]]).is_none());
        assert!(subdivide(&square(), &[vec![[1.0, 1.0], [3.0, 3.0]]]).is_none());
    }

    #[test]
    fn a_degenerate_chord_is_refused() {
        assert!(subdivide(&square(), &[vec![[0.0, 2.0]]]).is_none());
        // Both ends at the same boundary point.
        assert!(subdivide(&square(), &[vec![[0.0, 2.0], [0.0, 2.0]]]).is_none());
    }

    #[test]
    fn crossing_chords_give_four_pieces() {
        // What two boxes overlapping at a corner put on the face between them:
        // one cut from each of the other solid's side faces, meeting in the
        // middle. Without the crossing as a node, one chord passes through the
        // other and the walk reads pieces that are not there.
        let r = subdivide(
            &square(),
            &[vec![[0.0, 2.0], [4.0, 2.0]], vec![[2.0, 0.0], [2.0, 4.0]]],
        )
        .unwrap();
        assert_eq!(r.len(), 4);
        assert_eq!(areas(&r), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn an_off_centre_crossing_gives_the_right_four() {
        let r = subdivide(
            &square(),
            &[vec![[0.0, 1.0], [4.0, 1.0]], vec![[3.0, 0.0], [3.0, 4.0]]],
        )
        .unwrap();
        assert_eq!(r.len(), 4);
        assert_eq!(areas(&r), vec![3.0, 3.0, 1.0, 9.0].tap_sort());
    }

    #[test]
    fn three_chords_crossing_pairwise() {
        // A triangle of cuts: three crossings, seven pieces.
        let r = subdivide(
            &square(),
            &[
                vec![[0.0, 1.0], [4.0, 1.0]],
                vec![[0.0, 3.0], [4.0, 3.0]],
                vec![[2.0, 0.0], [2.0, 4.0]],
            ],
        )
        .unwrap();
        assert_eq!(r.len(), 6);
        let total: f64 = r.iter().map(|x| x.area).sum();
        assert!((total - 16.0).abs() < 1e-9, "{:?}", areas(&r));
    }

    #[test]
    fn a_curved_chord_crossing_a_straight_one() {
        // The crossing is found on the sampled polyline, so a bowed cut works
        // the same way a straight one does.
        let bow: Vec<[f64; 2]> = (0..=16)
            .map(|i| {
                let t = i as f64 / 16.0;
                [4.0 * t, 2.0 + 1.5 * (std::f64::consts::PI * t).sin()]
            })
            .collect();
        let r = subdivide(&square(), &[bow, vec![[2.0, 0.0], [2.0, 4.0]]]).unwrap();
        assert_eq!(r.len(), 4);
        let total: f64 = r.iter().map(|x| x.area).sum();
        assert!((total - 16.0).abs() < 1e-9, "{:?}", areas(&r));
    }

    #[test]
    fn a_closed_path_inside_a_region_becomes_a_hole() {
        // A cut does not always separate a region in two. A closed curve inside
        // one leaves it connected, with a hole — which is what an intersection
        // curve that closes on a face is, and what a shared wall's outline is
        // when one face sits entirely inside the other.
        let ring: Vec<[f64; 2]> = (0..=12)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 12.0;
                [2.0 + t.cos(), 2.0 + t.sin()]
            })
            .collect();
        let r = subdivide(&square(), &[ring]).unwrap();
        assert_eq!(r.len(), 2, "the surround and the disc");

        let surround = r
            .iter()
            .find(|x| x.holes.len() == 1)
            .expect("one has a hole");
        let disc = r
            .iter()
            .find(|x| x.holes.is_empty())
            .expect("the other does not");
        // 16 less the unit disc, and the disc itself — as twelve-gons.
        assert!((surround.area + disc.area - 16.0).abs() < 1e-9);
        assert!(disc.area > 2.9 && disc.area < 3.15, "{}", disc.area);
    }

    #[test]
    fn a_hole_and_a_cut_together() {
        // Both at once, which is what a face gets when one solid both crosses it
        // and shares a wall with it.
        let ring: Vec<[f64; 2]> = (0..=12)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 12.0;
                [1.0 + 0.5 * t.cos(), 1.0 + 0.5 * t.sin()]
            })
            .collect();
        let r = subdivide(&square(), &[ring, vec![[0.0, 3.0], [4.0, 3.0]]]).unwrap();
        let total: f64 = r.iter().map(|x| x.area).sum();
        assert!((total - 16.0).abs() < 1e-9, "areas {:?}", areas(&r));
        assert_eq!(r.iter().filter(|x| !x.holes.is_empty()).count(), 1);
    }

    #[test]
    fn clipping_a_path_to_the_region() {
        // Entirely inside: unchanged.
        let inside = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]];
        let r = clip_to_region(&square(), &inside);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].len(), inside.len());

        // Entirely outside: nothing.
        let outside = vec![[5.0, 5.0], [7.0, 5.0], [7.0, 7.0], [5.0, 5.0]];
        assert!(clip_to_region(&square(), &outside).is_empty());

        // Straddling: one stretch, entering and leaving through the boundary,
        // which is exactly a chord.
        let across = vec![[2.0, -1.0], [2.0, 5.0]];
        let r = clip_to_region(&square(), &across);
        assert_eq!(r.len(), 1);
        assert!((r[0][0][1] - 0.0).abs() < 1e-9, "{:?}", r[0][0]);
        assert!((r[0][r[0].len() - 1][1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_partly_overlapping_square_cuts_this_one_in_two() {
        // The case that made a coincident-face rule impossible without this:
        // neither region contains the other. Clipping the second's outline to
        // the first gives a chord, and the subdivision does the rest — which
        // together is a polygon *boolean*, built out of the arrangement.
        let other = vec![[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0], [2.0, 2.0]];
        let chords = clip_to_region(&square(), &other);
        assert_eq!(chords.len(), 1, "one stretch of it is inside");

        let r = subdivide(&square(), &chords).unwrap();
        assert_eq!(r.len(), 2);
        // The overlap is 2×2; the rest of the 4×4 is 12.
        assert_eq!(areas(&r), vec![4.0, 12.0]);
    }

    #[test]
    fn an_overlap_that_crosses_two_sides() {
        // The other region cuts a corner off rather than a strip.
        let other = vec![
            [3.0, -1.0],
            [5.0, -1.0],
            [5.0, 5.0],
            [3.0, 5.0],
            [3.0, -1.0],
        ];
        let chords = clip_to_region(&square(), &other);
        let r = subdivide(&square(), &chords).unwrap();
        let total: f64 = r.iter().map(|x| x.area).sum();
        assert!((total - 16.0).abs() < 1e-9, "{:?}", areas(&r));
        assert_eq!(areas(&r), vec![4.0, 12.0]);
    }

    #[test]
    fn chords_that_are_not_a_clean_cut_are_refused() {
        // Along the boundary: doubles an edge and cuts nothing off.
        assert!(subdivide(&square(), &[vec![[0.0, 0.0], [0.0, 4.0]]]).is_none());
    }

    #[test]
    fn a_chord_may_end_on_another_chord() {
        // A T-junction, which is not a defect: two boxes overlapping at a corner
        // put two cuts on the face between them that stop at the corner they
        // share. Neither reaches the boundary at both ends, and together they
        // cross the face.
        let r = subdivide(
            &square(),
            &[vec![[1.0, 0.0], [1.0, 4.0]], vec![[0.0, 1.0], [1.0, 1.0]]],
        )
        .unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(areas(&r), vec![1.0, 3.0, 12.0]);
    }

    #[test]
    fn two_chords_meeting_end_to_end_cut_a_corner_off() {
        // Exactly the box-on-box case: an L of cuts across one corner.
        let r = subdivide(
            &square(),
            &[vec![[2.0, 0.0], [2.0, 2.0]], vec![[2.0, 2.0], [0.0, 2.0]]],
        )
        .unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(areas(&r), vec![4.0, 12.0]);
    }

    #[test]
    fn sources_point_back_at_the_inputs() {
        // Without this the caller cannot recover vertex indices, and a seam that
        // is merely coincident is not shared.
        let r = subdivide(&square(), &[vec![[0.0, 2.0], [4.0, 2.0]]]).unwrap();
        let mut boundary = 0;
        let mut crossing = 0;
        for region in &r {
            for s in &region.outer.sources {
                match s {
                    Source::Boundary(_) => boundary += 1,
                    Source::Crossing => crossing += 1,
                    Source::Chord { .. } => {}
                }
            }
        }
        assert_eq!(boundary, 4, "each original corner used once");
        assert_eq!(crossing, 4, "two new points, each used by two pieces");
    }

    trait TapSort {
        fn tap_sort(self) -> Self;
    }
    impl TapSort for Vec<f64> {
        fn tap_sort(mut self) -> Self {
            self.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self
        }
    }
}
