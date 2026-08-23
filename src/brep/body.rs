//! A body whose faces share **vertices**, not merely parameters.
//!
//! Stage 3b's first piece, and the answer to the one thing the mesh-side
//! re-tessellation in [`mod@super::retessellate`] cannot do.
//!
//! # Why sharing parameters is not enough
//!
//! Two faces meeting along a rim can agree perfectly about where that rim is —
//! both derive it from the same pair of surfaces — and still produce a cracked
//! mesh, because each evaluates the rim's points *itself*. The two arrive at
//! values a few ULPs apart by different arithmetic, and a positional weld with
//! any fixed tolerance either misses them or starts merging things it shouldn't.
//! Measured, on a cone: densifying a shared boundary and letting both faces
//! sample it took a closed result to 118 open edges.
//!
//! The fix is not a better tolerance. It is to tessellate each edge **once**,
//! into a list of vertex *indices*, and have both faces use those indices. Then
//! the faces are welded by construction and no tolerance is involved.
//!
//! # What this is not, yet
//!
//! Faces here are a surface plus the boundary edges around them. There are no
//! trim loops in parameter space, no p-curves, no tolerant modelling, and no
//! boolean. A face's interior is filled from its parameter footprint, which is
//! rectangular for the quadrics and bounded by the boundary for the planes. That
//! is enough to build a watertight mesh at any tolerance from a tagged solid,
//! which is what Stage 3a promised; it is not a general B-rep.

use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Matrix4;
use crate::nurbs::v3;
use crate::nurbs::V3;

use super::table::{triangle_normal, triangle_vertices};
use super::{Surface, SurfaceTable};

/// A boundary shared by exactly two faces, tessellated once.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The two surfaces meeting here.
    pub surfaces: (usize, usize),
    /// Vertex indices into [`Body::vertices`], in order.
    pub vertices: Vec<usize>,
    /// A closed loop repeats its first vertex last.
    pub closed: bool,
}

/// One surface patch, with the edges that bound it.
#[derive(Debug, Clone)]
pub struct Face {
    pub surface: usize,
    /// Indices into [`Body::edges`].
    pub edges: Vec<usize>,
    /// Does the mesh wind this face against the surface's outward normal?
    pub flipped: bool,
    /// The parameter footprint the face occupies, taken from the mesh.
    ///
    /// Needed because a face's extent is not derivable from its edges alone: a
    /// cone's side has *one* rim, and its other end is the apex — a place the
    /// surface degenerates rather than a boundary anything can be shared with.
    pub u_range: (f64, f64),
    pub v_range: (f64, f64),
    /// Does the face wrap the full turn in `u` / `v`? Then its grid closes on
    /// itself instead of leaving a seam. A sphere wraps in `u`; a torus in both.
    pub u_wraps: bool,
    pub v_wraps: bool,
    /// Trim loops given directly, rather than derived from [`Self::edges`].
    ///
    /// Derivation is right for a body recovered from a mesh or authored from
    /// primitives: the edges already determine the loops, and two
    /// representations of one fact drift apart. It is *not* enough for a boolean
    /// result, whose faces are bounded partly by original edges and partly by
    /// intersection curves that no edge of either input carries. Those faces
    /// state their loops.
    pub loops: Option<Vec<TrimLoop>>,
}

/// A solid as surfaces, shared edges and faces.
#[derive(Debug, Clone)]
pub struct Body {
    surfaces: Vec<Surface>,
    vertices: Vec<V3>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
}

/// What a tessellation produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyReport {
    pub faces: usize,
    pub edges: usize,
    pub vertices: usize,
    pub triangles: usize,
    /// Edges used by exactly one triangle. Zero means closed.
    pub boundary_edges: usize,
    /// Faces whose interior could not be filled; their input triangles are
    /// carried through unchanged.
    pub carried_through: usize,
}

impl BodyReport {
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0
    }
}

impl Body {
    /// Assemble from parts. Used by the boolean, which builds faces directly
    /// rather than deriving them from a mesh.
    pub(crate) fn from_parts(
        surfaces: Vec<Surface>,
        vertices: Vec<V3>,
        edges: Vec<Edge>,
        faces: Vec<Face>,
    ) -> Body {
        Body {
            surfaces,
            vertices,
            edges,
            faces,
        }
    }

    /// Mutable face access, for tests that need to build a deliberately
    /// malformed body.
    #[cfg(test)]
    pub(crate) fn faces_mut(&mut self) -> &mut Vec<Face> {
        &mut self.faces
    }

    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }
    pub fn vertices(&self) -> &[V3] {
        &self.vertices
    }
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    /// Build a body from a mesh carrying surface provenance.
    ///
    /// Faces come from the provenance; edges come from
    /// [`super::stitch::patch_boundaries`], which finds them by looking for mesh
    /// edges whose two triangles disagree about their surface. Without
    /// provenance that question has no answer, which is why this cannot be done
    /// from a triangle soup.
    pub fn from_tagged_mesh(geometry: &BufferGeometry) -> Option<Body> {
        let table: &SurfaceTable = geometry.surface_table()?;
        let boundaries = super::stitch::patch_boundaries(geometry, table);

        // Global vertex pool, deduplicated by position so an endpoint shared by
        // several edges is one vertex.
        let extent = {
            let pos = geometry.get_attribute("position")?;
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &c in &pos.array {
                lo = lo.min(c as f64);
                hi = hi.max(c as f64);
            }
            if lo.is_finite() {
                (hi - lo).max(1.0)
            } else {
                1.0
            }
        };
        let quantum = extent * 1e-6;
        let mut pool: HashMap<[i64; 3], usize> = HashMap::new();
        let mut vertices: Vec<V3> = Vec::new();
        let mut intern = |p: V3, vertices: &mut Vec<V3>| -> usize {
            let key = [
                (p[0] / quantum).round() as i64,
                (p[1] / quantum).round() as i64,
                (p[2] / quantum).round() as i64,
            ];
            *pool.entry(key).or_insert_with(|| {
                vertices.push(p);
                vertices.len() - 1
            })
        };

        let edges: Vec<Edge> = boundaries
            .iter()
            .map(|b| Edge {
                surfaces: b.surfaces,
                vertices: b.points.iter().map(|p| intern(*p, &mut vertices)).collect(),
                closed: b.closed,
            })
            .collect();

        // One face per surface that owns triangles, with its winding taken from
        // the mesh — a subtracted cavity winds the other way and must keep doing so.
        let faces = table
            .groups()
            .into_iter()
            .map(|(si, tris)| {
                let surface = &table.surfaces()[si];
                let mut votes = 0i32;
                for tri in &tris {
                    let Some(v) = triangle_vertices(geometry, *tri) else {
                        continue;
                    };
                    if let (Some(face), Some((u, w))) =
                        (triangle_normal(&v), surface.invert(centroid(&v)))
                    {
                        if let Some(sn) = surface.normal(u, w) {
                            votes += if v3::dot(face, sn) < 0.0 { -1 } else { 1 };
                        }
                    }
                }
                // Parameter footprint, from the mesh's own vertices.
                let (mut u0, mut u1) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY);
                for tri in &tris {
                    let Some(v) = triangle_vertices(geometry, *tri) else {
                        continue;
                    };
                    for p in v {
                        if let Some((pu, pv)) = surface.invert(p) {
                            u0 = u0.min(pu);
                            u1 = u1.max(pu);
                            v0 = v0.min(pv);
                            v1 = v1.max(pv);
                        }
                    }
                }
                let (pu, pv) = surface.periodic();
                // A full turn shows up as a range approaching 2π — it stops just
                // short because there is no vertex *at* the branch cut.
                let full = std::f64::consts::TAU * 0.75;
                let u_wraps = pu && (u1 - u0) > full;
                let v_wraps = pv && (v1 - v0) > full;

                Face {
                    surface: si,
                    edges: (0..edges.len())
                        .filter(|&e| edges[e].surfaces.0 == si || edges[e].surfaces.1 == si)
                        .collect(),
                    flipped: votes < 0,
                    u_range: (u0, u1),
                    v_range: (v0, v1),
                    u_wraps,
                    v_wraps,
                    loops: None,
                }
            })
            .collect();

        Some(Body {
            surfaces: table.surfaces().to_vec(),
            vertices,
            edges,
            faces,
        })
    }

    /// Refine every edge until its chords are within `tolerance` of both
    /// surfaces.
    ///
    /// Done here, once, on the shared vertex list — which is the whole point.
    /// The same refinement performed independently by each face is what produced
    /// the cracks this type exists to avoid.
    pub fn refine_edges(&mut self, tolerance: f64) {
        // An edge a *trimmed* face also describes is left alone.
        //
        // This used to be because the face's loop was a fixed polyline sampled
        // where the edge happened to be, so refining one without the other put
        // them a point apart along a seam they were meant to hold shut. That is
        // no longer true: a loop is this face's edges in an order, every step of
        // every ring is a step of some edge, and the walk that reads one off the
        // other now succeeds on every face measured.
        //
        // So the ring *can* be rebuilt from the same walk once the edges have
        // moved, and that was built and measured. The walk is exact: on every
        // ring of a bored ball the concatenation reproduces the ring
        // identically, seamed ring included, vertices and — once the parameter
        // anchor is seeded from where the ring already put its first point —
        // parameters too. Rebuilding every face except those that walk an edge
        // twice leaves the whole suite green.
        //
        // It is still not here, and a second run at it says the same. The hope
        // was that `split_face` could then stop dropping a face's rings — it
        // drops them to stop them diverging from the edges, and the sphere that
        // loses three of them cannot be filled at all without them, which is 407
        // open edges around a face that is not there. Rebuild them from the
        // edges and the divergence has nowhere to come from, so keeping them
        // should be free.
        //
        // It is not. Built again — the walk reads, the whole suite stays green,
        // the corpus is unchanged, a bored plate's volume converges to the same
        // six figures at every tolerance — and keeping the rings still costs
        // eight boolean tests and a STEP test. The reason is the one below:
        // the rings that get kept are the ones the walk *cannot* read, being
        // parameter outlines no edge carries, so they stay pinned and coarse
        // exactly as before. The rebuild does not reach them.
        //
        // So the prerequisite is not this walk. It is the sentence after it: a
        // complement face's outline is the last part of a boundary still not
        // made of edges.
        //
        // Half of that is now done — `materialise_seam_steps` gives a swept
        // face's seam an edge, taking a bored rod's rings from two unbacked
        // steps to none — and it does not move this. Keeping the rings still
        // costs nine boolean tests, and one of the nine is
        // `every_step_of_a_ring_is_a_step_of_an_edge`, which says why in as many
        // words: the rings that get kept are the unbacked ones. What is left
        // unbacked anywhere measured is a *pole* — a ring stepping from a vertex
        // to itself — and no edge can be made of that, since there is no line
        // there. Whatever fixes it is not another edge. The faces that would gain are curved ones stating rings, and
        // of those:
        //
        //   * a bored ball and a bored plate state no rings at all, so the pin
        //     never touched them;
        //   * two balls cut apart state a ring that is the *parameter outline* —
        //     seam and poles — which no edge carries, so the walk fails and the
        //     edges stay pinned anyway. Loop sizes [64, 32] and chord 0.392 at
        //     every tolerance from 1e-2 to 1e-4;
        //   * the rings that do walk cleanly are planar, and subdividing a
        //     straight edge adds nothing.
        //
        // So the obstacle is no longer the pin. It is that a complement face's
        // outline is the last part of a boundary still not made of edges. Until
        // that is, unpinning buys precision no face can spend.
        //
        // The reach also argues for waiting: the boolean refines its own result
        // before checking closure (`boolean.rs:819`), so this decides what the
        // kernel accepts, not only what a caller tessellates.
        // Nothing is pinned: `restate_rings` puts each ring back in the edges'
        // own vertices afterwards, so a ring cannot be left behind by an edge
        // that moved. Pinned, a drill's end cap kept a 16-point rim beside a
        // wall of 129 and the blind hole came back 1.3% small.
        let pinned: Vec<bool> = vec![false; self.edges.len()];
        for (index, e) in self.edges.iter_mut().enumerate() {
            if pinned[index] {
                continue;
            }
            let (a, b) = (&self.surfaces[e.surfaces.0], &self.surfaces[e.surfaces.1]);
            let mut out: Vec<usize> = Vec::with_capacity(e.vertices.len());
            for w in e.vertices.windows(2) {
                out.push(w[0]);
                let (p, q) = (self.vertices[w[0]], self.vertices[w[1]]);
                subdivide(p, q, a, b, tolerance, 0, &mut self.vertices, &mut out);
            }
            if let Some(&last) = e.vertices.last() {
                out.push(last);
            }
            e.vertices = out;
        }
        self.restate_rings();
    }

    /// Write every ring in terms of the edges it runs along.
    ///
    /// A ring is a fixed polyline and the edges beside it are not: a face that
    /// arrives already sampled keeps that sampling, and where the boolean
    /// re-sampled the boundary the two disagree. A rod joined to a bore across
    /// it comes back with the bore's end caps holding sixteen-point rims and the
    /// wall between them holding seventy-eight, and the six edges where the
    /// cap's ring cuts a corner off the wall's are open.
    ///
    /// Matching by *endpoint* rather than by adjacency is what makes this work.
    /// The cap's step from one rim vertex to another is a step of the rim, five
    /// of its points at a time; taking the path between them restates the cap's
    /// ring in the wall's own vertices, and then they cannot disagree.
    pub(crate) fn restate_rings(&mut self) {
        let mut on: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (ei, e) in self.edges.iter().enumerate() {
            for (k, v) in e.vertices.iter().enumerate() {
                on.entry(*v).or_default().push((ei, k));
            }
        }
        let mut done: Vec<(usize, Vec<TrimLoop>)> = Vec::new();
        for (fi, face) in self.faces.iter().enumerate() {
            let Some(loops) = face.loops.as_ref().filter(|l| !l.is_empty()) else {
                continue;
            };
            let surface = &self.surfaces[face.surface];
            let (pu, pv) = surface.periodic();
            let mut out = Vec::with_capacity(loops.len());
            for l in loops {
                let n = l.vertices.len();
                if n < 3 || l.uv.len() != n {
                    out.push(l.clone());
                    continue;
                }
                let mut uv: Vec<[f64; 2]> = Vec::with_capacity(n);
                let mut vs: Vec<usize> = Vec::with_capacity(n);
                for i in 0..n {
                    let (a, b) = (l.vertices[i], l.vertices[(i + 1) % n]);
                    vs.push(a);
                    uv.push(l.uv[i]);
                    if a == b {
                        continue;
                    }
                    // The shortest run of one edge that joins them.
                    let mut best: Option<Vec<usize>> = None;
                    for &(ea, ka) in on.get(&a).into_iter().flatten() {
                        for &(eb, kb) in on.get(&b).into_iter().flatten() {
                            if ea != eb || ka == kb {
                                continue;
                            }
                            let path: Vec<usize> = if ka < kb {
                                self.edges[ea].vertices[ka + 1..kb].to_vec()
                            } else {
                                self.edges[ea].vertices[kb + 1..ka]
                                    .iter()
                                    .rev()
                                    .copied()
                                    .collect()
                            };
                            let mut walked = 0.0;
                            let mut at = self.vertices[a];
                            for &v in path.iter().chain(std::iter::once(&b)) {
                                walked += v3::dist(at, self.vertices[v]);
                                at = self.vertices[v];
                            }
                            if walked > v3::dist(self.vertices[a], self.vertices[b]) * 1.5 + 1e-9 {
                                continue;
                            }
                            if best.as_ref().is_none_or(|w| path.len() < w.len()) {
                                best = Some(path);
                            }
                        }
                    }
                    // Unwrapped along the run, and only kept if the run ends
                    // where the ring's own next point already is.
                    //
                    // The ring's points are left exactly as they are: their
                    // parameters carry the orientation the rest of the kernel
                    // reads, and rewriting them flips rings from outer to hole.
                    // So the inserted points have to meet them, and a run that
                    // does not is a run unwrapped the wrong way round — it folds
                    // the ring, and a drill wall's rim lost two thirds of its
                    // area that way.
                    let mut anchor = l.uv[i];
                    let mut run: Vec<(usize, [f64; 2])> = Vec::new();
                    for v in best.into_iter().flatten() {
                        let Some((u, w)) = surface.invert(self.vertices[v]) else {
                            continue;
                        };
                        let p = [
                            if pu { near(u, anchor[0]) } else { u },
                            if pv { near(w, anchor[1]) } else { w },
                        ];
                        run.push((v, p));
                        anchor = p;
                    }
                    // A run that does not land is not wrong, only wound the
                    // other way: shift the whole of it by whole turns until its
                    // end meets the ring's next point. Dropping it instead left
                    // the two faces on one rim holding 29 points and 25.
                    if let Some((_, last)) = run.last().copied() {
                        let want = l.uv[(i + 1) % n];
                        let shift = [
                            if pu {
                                near(last[0], want[0]) - last[0]
                            } else {
                                0.0
                            },
                            if pv {
                                near(last[1], want[1]) - last[1]
                            } else {
                                0.0
                            },
                        ];
                        for (_, p) in run.iter_mut() {
                            p[0] += shift[0];
                            p[1] += shift[1];
                        }
                    }
                    for (v, p) in run {
                        uv.push(p);
                        vs.push(v);
                    }
                }
                let fresh = TrimLoop {
                    area: signed_area(&uv),
                    uv,
                    vertices: vs,
                };
                out.push(fresh);
            }
            done.push((fi, out));
        }
        for (fi, loops) in done {
            self.faces[fi].loops = Some(loops);
        }
    }

    /// Tessellate to a `BufferGeometry`.
    ///
    /// Edge vertices are emitted once and referenced by both adjacent faces, so
    /// the result is closed by construction rather than by welding.
    /// A face with no stored loops goes to the grid whatever its edges say, and
    /// a grid over a parameter rectangle cannot describe a sphere with holes in
    /// it. `ball - cross` hands its sphere three rings — 130, 63, 63 — and after
    /// a bore is cut through it the same face comes back with `loops: None`, so
    /// the grid takes it, fails, and the face is dropped: one `carried_through`,
    /// and the four faces around it left holding 407 open edges against nothing.
    ///
    /// Falling back to `fill_planar`, which derives its rings from the
    /// edges, does fill it — 407 open edges to 288, 142 to 55 elsewhere — but
    /// flips no case in a 45-chain corpus and fires nowhere in this crate's
    /// tests, so there is nothing to hold it in place and it is not here. The
    /// loops are the thing to fix: they exist before that second cut and not
    /// after it.
    pub fn tessellate(&self, tolerance: f64) -> (BufferGeometry, BodyReport) {
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Every body vertex is emitted up front, so an edge's vertices have one
        // index that both its faces use.
        let mut emitted: Vec<u32> = Vec::with_capacity(self.vertices.len());
        for p in &self.vertices {
            emitted.push((positions.len() / 3) as u32);
            positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
            normals.extend_from_slice(&[0.0, 0.0, 0.0]); // filled per face below
        }

        let mut carried_through = 0usize;
        let mut triangle_face: Vec<u32> = Vec::new();
        for (face_index, face) in self.faces.iter().enumerate() {
            let surface = &self.surfaces[face.surface];
            let before = indices.len();
            // A face that carries trim loops is described by them, whatever it
            // lies on: a cylinder cut by a crossing bore has an outline no
            // parameter rectangle covers. The loop fill handles any surface —
            // it works in `(u, v)` throughout — and refines the interior to the
            // surface afterwards.
            let trimmed = matches!(face.footprint(), Footprint::Rings(_));
            let ok = match surface {
                _ if trimmed => self.fill_planar(
                    face_index,
                    surface,
                    tolerance,
                    &emitted,
                    &mut positions,
                    &mut normals,
                    &mut indices,
                ),
                Surface::Plane { .. } => self.fill_planar(
                    face_index,
                    surface,
                    tolerance,
                    &emitted,
                    &mut positions,
                    &mut normals,
                    &mut indices,
                ),
                _ => {
                    // Try the swept orientation, then the transposed one. A
                    // failed attempt may have emitted rows before it gave up, so
                    // roll back to where it started rather than leaving them.
                    let mark = (positions.len(), normals.len(), indices.len());
                    self.fill_parametric(
                        face,
                        surface,
                        tolerance,
                        false,
                        &emitted,
                        &mut positions,
                        &mut normals,
                        &mut indices,
                    ) || {
                        positions.truncate(mark.0);
                        normals.truncate(mark.1);
                        indices.truncate(mark.2);
                        self.fill_parametric(
                            face,
                            surface,
                            tolerance,
                            true,
                            &emitted,
                            &mut positions,
                            &mut normals,
                            &mut indices,
                        )
                    }
                }
            };
            if !ok {
                carried_through += 1;
                continue;
            }
            // One entry per *triangle*, not per index. `orient` looks this up
            // by triangle number, so three entries each put every triangle but
            // the first under some other face's normal — and the majority vote
            // that decides which way a shell faces was reading the wrong ones.
            for _ in (before..indices.len()).step_by(3) {
                triangle_face.push(face_index as u32);
            }
        }
        orient(
            &self.surfaces,
            &self.faces,
            &triangle_face,
            &positions,
            &mut indices,
        );

        // Normals for the shared vertices: analytic, from whichever face owns
        // them. A vertex on an edge belongs to two faces with different normals;
        // taking one is what a shared-vertex mesh means, and splitting them is a
        // rendering decision the caller can make later.
        for (vi, p) in self.vertices.iter().enumerate() {
            let mut n = [0.0f64, 0.0, 1.0];
            for face in &self.faces {
                let s = &self.surfaces[face.surface];
                if s.distance(*p) > 1e-6 {
                    continue;
                }
                if let Some((u, v)) = s.invert(*p) {
                    if let Some(sn) = s.normal(u, v) {
                        n = if face.flipped {
                            v3::scale(sn, -1.0)
                        } else {
                            sn
                        };
                        break;
                    }
                }
            }
            let o = emitted[vi] as usize * 3;
            normals[o] = n[0] as f32;
            normals[o + 1] = n[1] as f32;
            normals[o + 2] = n[2] as f32;
        }

        let boundary_edges = count_boundary_edges(&indices);
        let mut g = BufferGeometry::new();
        let vertices = positions.len() / 3;
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_index(indices.clone());

        (
            g,
            BodyReport {
                faces: self.faces.len(),
                edges: self.edges.len(),
                vertices,
                triangles: indices.len() / 3,
                boundary_edges,
                carried_through,
            },
        )
    }

    /// Fill a planar face by triangulating its trim loops.
    ///
    /// A plane is exact under any triangulation, so the loops *are* the answer —
    /// no interior samples, and nothing outside the face. Holes are bridged into
    /// the outer ring first, which is what makes a face with a hole come back
    /// with one instead of filled solid.
    #[allow(clippy::too_many_arguments)]
    fn fill_planar(
        &self,
        face_index: usize,
        surface: &Surface,
        tolerance: f64,
        emitted: &[u32],
        positions: &mut Vec<f32>,
        normals: &mut Vec<f32>,
        indices: &mut Vec<u32>,
    ) -> bool {
        let face = &self.faces[face_index];
        let loops = self.face_loops(face_index);
        if loops.is_empty() {
            return false;
        }
        let Some(outer) = loops.iter().find(|l| l.is_outer()) else {
            return false;
        };
        let holes: Vec<&TrimLoop> = loops.iter().filter(|l| l.is_hole()).collect();

        // A ring that runs its own loop twice is one loop.
        //
        // Measured on the boolean suite: a hole of 142 points where every one of
        // the 71 in its first half is the same *place* as the one 71 later. Both
        // passes go the same way round, so the areas add instead of cancelling
        // and it reads as a perfectly good hole of twice the size — while
        // crossing itself 124 times, which leaves the face two thirds filled.
        //
        // Halving it is decided by the ring's own points, so two faces sharing
        // the ring both halve it and neither is left describing it differently.
        let undoubled = |uv: &Vec<[f64; 2]>| -> Vec<[f64; 2]> {
            let n = uv.len();
            if n < 8 || !n.is_multiple_of(2) {
                return uv.clone();
            }
            let half = n / 2;
            let doubled = (0..half).all(|k| {
                let a = surface.point(uv[k][0], uv[k][1]);
                let b = surface.point(uv[k + half][0], uv[k + half][1]);
                v3::dist(a, b) <= tolerance
            });
            if doubled {
                uv[..half].to_vec()
            } else {
                uv.clone()
            }
        };

        // One simple polygon covering the face, holes spliced in.
        let hole_uv: Vec<Vec<[f64; 2]>> = holes.iter().map(|h| undoubled(&h.uv)).collect();
        let Some((ring_uv, ring_map)) = bridge_holes(&outer.uv, &hole_uv) else {
            if std::env::var("THREERS_BREP_DEBUG").is_ok() {
                eprintln!("  [brep] face {face_index}: bridge_holes failed");
            }
            return false;
        };

        // `ring_map` indexes the concatenation outer ++ holes. Each entry
        // resolves to a *body* vertex when the loop was derived from edges, and
        // to a freshly evaluated one when it was not — a boolean result's loops
        // are intersection curves, which no edge of either input carries.
        let mut lookup: Vec<Option<usize>> = Vec::new();
        let mut lookup_uv: Vec<[f64; 2]> = Vec::new();
        let push_loop = |l: &TrimLoop, lookup: &mut Vec<Option<usize>>, uv: &mut Vec<[f64; 2]>| {
            for k in 0..l.uv.len() {
                lookup.push(l.vertices.get(k).copied());
                uv.push(l.uv[k]);
            }
        };
        push_loop(outer, &mut lookup, &mut lookup_uv);
        for (h, uv) in holes.iter().zip(&hole_uv) {
            if uv.len() == h.uv.len() {
                push_loop(h, &mut lookup, &mut lookup_uv);
            } else {
                // Halved: the points kept are the ring's own first pass, and
                // they keep the names they had.
                for (k, &texcoord) in uv.iter().enumerate() {
                    lookup.push(h.vertices.get(k).copied());
                    lookup_uv.push(texcoord);
                }
            }
        }

        let mut fresh: HashMap<usize, u32> = HashMap::new();
        let mut ring: Vec<u32> = Vec::with_capacity(ring_map.len());
        for &i in &ring_map {
            let Some(slot) = lookup.get(i) else { break };
            match slot.filter(|vi| *vi < emitted.len()) {
                Some(vi) => ring.push(emitted[vi]),
                None => {
                    let id = *fresh.entry(i).or_insert_with(|| {
                        let uv = lookup_uv[i];
                        let p = surface.point(uv[0], uv[1]);
                        let mut n = surface.normal(uv[0], uv[1]).unwrap_or([0.0, 0.0, 1.0]);
                        if face.flipped {
                            n = v3::scale(n, -1.0);
                        }
                        positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
                        normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);
                        (positions.len() / 3 - 1) as u32
                    });
                    ring.push(id);
                }
            }
        }
        if ring.len() != ring_uv.len() || ring.len() < 3 {
            if std::env::var("THREERS_BREP_DEBUG").is_ok() {
                eprintln!(
                    "  [brep] face {face_index}: ring {} vs uv {}",
                    ring.len(),
                    ring_uv.len()
                );
            }
            return false;
        }

        let tris = earclip(&ring_uv);

        // Say so when the clip did not cover the face.
        //
        // Ear clipping can stop with points unspent, and it did so silently: the
        // face came back part-filled and the only thing that noticed was the
        // watertight gate, counting edges it could not explain. A face that
        // could not be filled is `carried_through`, which is a fact the caller
        // can act on.
        //
        // Judged by *area*, not by points left over. The ring handed to the clip
        // is bridged, so a bridge is traversed twice and a leftover of sixty
        // points can enclose nothing at all — measured, two give-ups of `5 of
        // 71` that cover their rings completely. What matters is whether any of
        // the face is missing.
        {
            let want = signed_area(&ring_uv).abs();
            let got: f64 = tris
                .iter()
                .map(|t| {
                    let (a, b, c) = (ring_uv[t[0]], ring_uv[t[1]], ring_uv[t[2]]);
                    (((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0).abs()
                })
                .sum();
            if want > 0.0 && got < want * 0.99 {
                if std::env::var("THREERS_BREP_DEBUG").is_ok() {
                    eprintln!(
                        "  [brep] face {face_index}: filled {:.1}% of its ring",
                        got / want * 100.0
                    );
                }
                return false;
            }
        }

        // Ear clipping fills the *boundary*; on a curved surface that leaves the
        // interior spanning it in flat sheets. Refine until it does not.
        let tris = refine_on_surface(
            surface,
            face.flipped,
            &ring_uv,
            &ring,
            &tris,
            tolerance,
            positions,
            normals,
        );
        for t in &tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            // Only a triangle that names one vertex twice is dropped. A flat one
            // with three distinct corners still carries three distinct edges,
            // and those edges are how the neighbouring face stays attached.
            if a == b || b == c || c == a {
                continue;
            }
            if face.flipped {
                indices.extend_from_slice(&[a, c, b]);
            } else {
                indices.extend_from_slice(&[a, b, c]);
            }
        }
        true
    }

    /// Fill a curved face over its parameter rectangle, using the boundary's own
    /// vertices for the rows they lie on.
    ///
    /// A quadric patch's footprint *is* a rectangle in its parameters, so a grid
    /// is right — but the rows that coincide with a shared edge must be that
    /// edge's vertices, not freshly evaluated copies of them. That is the whole
    /// difference between this and re-meshing each patch on its own.
    ///
    /// Two cases the edges alone cannot express, and the stored footprint can:
    ///
    /// * **A degenerate end.** A cone's side has one rim; its other end is the
    ///   apex, which is not a boundary shared with anything. It becomes a single
    ///   vertex and the adjacent quads collapse to a fan.
    /// * **A closed sweep.** A full revolution's grid must join its last column
    ///   back to its first, or the result is open along the seam — which is
    ///   exactly the four stray edges this had before the footprint was recorded.
    #[allow(clippy::too_many_arguments)]
    fn fill_parametric(
        &self,
        face: &Face,
        surface: &Surface,
        tolerance: f64,
        transpose: bool,
        emitted: &[u32],
        positions: &mut Vec<f32>,
        normals: &mut Vec<f32>,
        indices: &mut Vec<u32>,
    ) -> bool {
        // The walk is written for rims at constant `v` — the swept case, where a
        // face is a band between two rings. `transpose` runs the same walk with
        // the two parameters exchanged, for a face bounded by *meridians*
        // instead: one half of a seam-split sphere, whose only edges are the two
        // constant-`u` cuts that gave it a boundary at all.
        let at = |a: f64, b: f64| {
            if transpose {
                surface.point(b, a)
            } else {
                surface.point(a, b)
            }
        };
        let nat = |a: f64, b: f64| {
            if transpose {
                surface.normal(b, a)
            } else {
                surface.normal(a, b)
            }
        };
        let inv = |p: V3| {
            surface
                .invert(p)
                .map(|(u, v)| if transpose { (v, u) } else { (u, v) })
        };
        let (a_range, b_range) = if transpose {
            (face.v_range, face.u_range)
        } else {
            (face.u_range, face.v_range)
        };
        let (a_wraps, b_wraps) = if transpose {
            (face.v_wraps, face.u_wraps)
        } else {
            (face.u_wraps, face.v_wraps)
        };

        let (pu, pv) = surface.periodic();
        let (a_periodic, b_periodic) = if transpose { (pv, pu) } else { (pu, pv) };

        // Each bounding edge of a swept face lies at a constant `v`.
        let mut rims: Vec<(f64, Vec<usize>)> = Vec::new();
        for &e in &face.edges {
            let edge = &self.edges[e];
            let mut params: Vec<(f64, f64, usize)> = Vec::with_capacity(edge.vertices.len());
            for &vi in &edge.vertices {
                let Some((u, v)) = inv(self.vertices[vi]) else {
                    return false;
                };
                params.push((u, v, vi));
            }
            // A pole contributes no opinion about where the rim sits: the
            // parameter is degenerate there, so `invert` returns an arbitrary
            // value. Two such points at the ends of a meridian widened its
            // measured spread from nothing to a half turn, and the fill then
            // refused a face it was about to succeed on.
            let solid: Vec<&(f64, f64, usize)> = params
                .iter()
                .filter(|p| !surface.parameter_degenerate_at(self.vertices[p.2]))
                .collect();
            let measured = if solid.len() >= 2 { &solid[..] } else { &[] };
            let vs: Vec<f64> = measured.iter().map(|p| p.1).collect();
            let us: Vec<f64> = measured.iter().map(|p| p.0).collect();
            // Measured the long way round, because a rim can sit *on* the branch
            // cut. A seam-split sphere's meridian is at u = ±π, where `invert`
            // returns whichever sign the arithmetic lands on, so the plain
            // min/max spread of a constant rim reads as a full turn and the fill
            // rejects the face it was about to succeed on.
            if vs.is_empty() {
                return false; // nothing but poles: no rim to read
            }
            let (v_mid, v_spread) = extent_of(&vs, b_periodic);
            let (_, u_spread) = extent_of(&us, a_periodic);
            if v_spread > u_spread.max(1e-12) * 0.1 {
                return false;
            }
            params.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            params.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);
            rims.push((v_mid, params.into_iter().map(|p| p.2).collect()));
        }
        if rims.len() > 2 {
            return false; // a face this simple fill does not describe
        }
        rims.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Angular samples. A rim supplies them when there is one — every row then
        // lines up with it. A boundary-less face (a sphere, a torus, an open
        // cylinder) has no rim to take them from and samples its own footprint.
        let column_u: Vec<f64> = match rims.first() {
            Some((_, verts)) => {
                let us: Vec<f64> = verts
                    .iter()
                    .filter_map(|&vi| inv(self.vertices[vi]).map(|(u, _)| u))
                    .collect();
                if us.len() != verts.len() {
                    return false;
                }
                us
            }
            None => {
                let (u0, u1) = a_range;
                let n = adaptive_steps(
                    tolerance,
                    |a, b| {
                        let m = 0.5 * (a + b);
                        let v = 0.5 * (b_range.0 + b_range.1);
                        v3::dist(at(m, v), v3::scale(v3::add(at(a, v), at(b, v)), 0.5))
                    },
                    u0,
                    u1,
                );
                // A full turn has no vertex at the branch cut, so the last
                // column stops one step short and the wrap closes the gap.
                let columns = if a_wraps { n } else { n + 1 };
                (0..columns)
                    .map(|i| u0 + (u1 - u0) * i as f64 / n as f64)
                    .collect()
            }
        };
        let n_u = column_u.len();
        if n_u < 3 {
            return false;
        }

        // Rows over the footprint.
        let (fv0, fv1) = b_range;
        if (fv1 - fv0).abs() < 1e-12 {
            return false;
        }
        let steps = adaptive_steps(
            tolerance,
            |a, b| {
                let mut worst = 0.0f64;
                for &u in column_u.iter().take(8) {
                    worst = worst.max(v3::dist(
                        at(u, 0.5 * (a + b)),
                        v3::scale(v3::add(at(u, a), at(u, b)), 0.5),
                    ));
                }
                worst
            },
            fv0,
            fv1,
        );
        let rows = if b_wraps { steps } else { steps + 1 };
        let row_v: Vec<f64> = (0..rows)
            .map(|i| fv0 + (fv1 - fv0) * i as f64 / steps as f64)
            .collect();

        let near = |a: f64, b: f64| {
            let mut d = (a - b).abs();
            if b_periodic {
                let period = std::f64::consts::TAU;
                d = d.min((period - d).abs());
            }
            d <= (fv1 - fv0).abs() * 1e-3
        };
        // Rows may differ in width: a rim is used verbatim however many points it
        // has, and the strip walk below joins rows that do not match.
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity(row_v.len());
        let mut grid_u: Vec<Vec<f64>> = Vec::with_capacity(row_v.len());
        for &v in &row_v {
            // A rim at this row? Use its vertices verbatim — that is the sharing.
            if let Some(rim) = rims.iter().find(|(rv, _)| near(*rv, v)) {
                let us: Vec<f64> = rim
                    .1
                    .iter()
                    .filter_map(|&vi| inv(self.vertices[vi]).map(|(u, _)| u))
                    .collect();
                if us.len() == rim.1.len() && us.len() >= 3 {
                    grid.push(rim.1.iter().map(|&vi| emitted[vi]).collect());
                    grid_u.push(us);
                    continue;
                }
            }
            // A degenerate row — a sphere's pole, a cone's apex — is one vertex,
            // and the quads against it collapse to a fan.
            let collapsed = {
                let a = at(column_u[0], v);
                let b = at(column_u[n_u / 2], v);
                v3::dist(a, b) < tolerance * 1e-3
            };
            let mut push = |u: f64| -> u32 {
                let p = at(u, v);
                let mut n = nat(u, v).unwrap_or([0.0, 0.0, 1.0]);
                if face.flipped {
                    n = v3::scale(n, -1.0);
                }
                let id = (positions.len() / 3) as u32;
                positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
                normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);
                id
            };
            if collapsed {
                let id = push(column_u[0]);
                grid.push(vec![id; n_u]);
            } else {
                grid.push(column_u.iter().map(|&u| push(u)).collect());
            }
            grid_u.push(column_u.clone());
        }

        // A degenerate *column* end — the pole of a sphere walked meridian-wise
        // — collapses the same way a degenerate row does, but the row check
        // above cannot see it: from this direction the pole is where every row
        // ends, not a row of its own. Left alone each row mints its own pole and
        // the fan cracks into as many slivers as there are rows.
        for end in [0usize, 1] {
            let col = |row: &Vec<u32>| if end == 0 { 0 } else { row.len() - 1 };
            let param = |row: &Vec<f64>| if end == 0 { row[0] } else { row[row.len() - 1] };
            if grid.iter().any(|r| r.is_empty()) {
                break;
            }
            let anchor = at(param(&grid_u[0]), row_v[0]);
            let same = grid_u
                .iter()
                .zip(&row_v)
                .all(|(us, &v)| v3::dist(at(param(us), v), anchor) < tolerance * 1e-3);
            if !same {
                continue;
            }
            let shared = grid[0][col(&grid[0])];
            for row in grid.iter_mut() {
                let c = col(row);
                row[c] = shared;
            }
        }

        // A full sweep joins its last row back to its first.
        let last_row = if b_wraps { grid.len() } else { grid.len() - 1 };
        for r in 0..last_row {
            let r2 = (r + 1) % grid.len();
            strip(
                &grid[r],
                &grid_u[r],
                &grid[r2],
                &grid_u[r2],
                a_wraps,
                face.flipped,
                positions,
                indices,
            );
        }
        true
    }
}

/// Refine a triangulation until it follows the surface it lies on.
///
/// Ear clipping fills a *boundary*. On a plane that is the whole answer; on a
/// cylinder or a torus it leaves the interior spanned by flat sheets, and a face
/// trimmed to an arbitrary outline has no parameter rectangle to grid instead.
///
/// The split is decided **per edge**, from the sagitta between its endpoints —
/// so two triangles sharing an edge always reach the same verdict and the mesh
/// cannot come apart along it. Boundary edges are never split: they are shared
/// with the neighbouring *face*, which is not party to this decision, and
/// splitting one here would open the seam.
#[allow(clippy::too_many_arguments)]
fn refine_on_surface(
    surface: &Surface,
    flipped: bool,
    ring_uv: &[[f64; 2]],
    ring: &[u32],
    tris: &[[usize; 3]],
    tolerance: f64,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
) -> Vec<[u32; 3]> {
    let mut uv: Vec<[f64; 2]> = ring_uv.to_vec();
    let mut id: Vec<u32> = ring.to_vec();
    let mut faces: Vec<[usize; 3]> = tris.to_vec();
    let n = ring_uv.len();
    // Consecutive in the ring means it *is* the boundary.
    //
    // So does a *chord* of the boundary whose midpoint lands back on it. Two
    // points of a rim sit at the same `v`, so the midpoint between any two of
    // them is at that `v` too — on the rim, geometrically, however far apart
    // along it they are. Splitting there puts a vertex on the boundary that the
    // face beyond it does not have, and the seam opens by however many such
    // splits happened.
    // How close to the ring counts as on it. Scaled to the face, and worked out
    // once: it does not depend on the point being tested, and recomputing it
    // inside `on_ring` walked the whole ring again on every call — which is
    // every edge of every triangle, every round of refinement.
    let near = {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for q in ring_uv {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12) * 1e-9
    };
    let on_ring = |p: [f64; 2]| {
        (0..n).any(|i| {
            let (x, y) = (ring_uv[i], ring_uv[(i + 1) % n]);
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
    // Whether an edge of the triangulation is an edge of the *face*.
    //
    // This is where a trimmed curved face loses its accuracy, and it is worth
    // being precise about why. Ear clipping fans, so it produces triangles with
    // an edge running from one ring vertex to another far along the same ring —
    // and on a cylinder every point of a rim shares that rim's `v`, so such a
    // chord lies on the boundary in parameter space however long it is, and is
    // refused here. It is refused for a real reason: splitting it puts a vertex
    // on the boundary that the face across the seam does not have.
    //
    // The cost is measurable. On a quarter cylinder at `1e-5` the worst such
    // triangle spans 83 degrees of a 90-degree wall and sags 0.75 off a radius
    // of 3 — 75,000 times the tolerance — and that is the whole of the half a
    // percent of volume the intersection is short. Splitting the triangle at
    // its centroid does not help: the three children keep all three original
    // edges, so the volume comes back bit for bit the same. The chord itself
    // has to go, which means not creating it — an edge flip against the
    // neighbouring triangle, or a triangulation of the parameter rectangle
    // rather than a fan of the loop.
    let boundary = |a: usize, b: usize| {
        if a < n && b < n && ((a + 1) % n == b || (b + 1) % n == a) {
            return true;
        }
        if a < n && b < n {
            let mid = [
                0.5 * (ring_uv[a][0] + ring_uv[b][0]),
                0.5 * (ring_uv[a][1] + ring_uv[b][1]),
            ];
            return on_ring(mid);
        }
        false
    };

    // Put the fan's chords back onto the boundary before refining anything.
    //
    // Only on a curved face. A plane is exact under any triangulation, so a
    // chord across one is not wrong and there is nothing to put back — but the
    // search still walked the ring for every edge of every triangle, and a
    // drill's end caps are the finest rings in the model. Half of what a
    // cylinder cost to tessellate was this, looking for a fault planes cannot
    // have.
    if !matches!(surface, Surface::Plane { .. })
    //
    // Ear clipping fans, so it emits triangles with an edge running from one
    // ring vertex to another far along the same ring. On a cylinder every point
    // of a rim shares that rim's `v`, so such a chord lies along the boundary
    // however long it is — and it cannot be split, because the new vertex would
    // land on the boundary and the face across the seam has no such point. Left
    // alone it is a flat sheet across a curved wall, worth half a percent of a
    // quarter cylinder's volume, and no tolerance moves it.
    //
    // The chord runs along the boundary, and the boundary already has vertices
    // between its ends. Use those: nothing is invented, every point of the fan
    // that replaces it is a ring vertex the face across the seam names too, so
    // the seam cannot open.
    //
    // Ear clipping also leaves *slivers* along such a chord — triangles with no
    // area in parameter space, whose corners are all on the ring. They are not
    // nothing. In three dimensions they are exactly the strip between the flat
    // chord and the curved boundary it shortcuts, and they are what was holding
    // that strip closed. Once the fan reaches the boundary the strip is covered
    // twice, so they go.
    {
        let flat = |t: &[usize; 3]| {
            ((uv[t[1]][0] - uv[t[0]][0]) * (uv[t[2]][1] - uv[t[0]][1])
                - (uv[t[2]][0] - uv[t[0]][0]) * (uv[t[1]][1] - uv[t[0]][1]))
                .abs()
                <= near * near
        };
        // The ring's own vertices between `a` and `b`, when the line between
        // them runs along the ring and the boundary bulges off it.
        let between = |a: usize, b: usize| -> Option<Vec<usize>> {
            if a >= n || b >= n || (a + 1) % n == b || (b + 1) % n == a {
                return None;
            }
            let (x, y) = (ring_uv[a], ring_uv[b]);
            let (dx, dy) = (y[0] - x[0], y[1] - x[1]);
            let len2 = dx * dx + dy * dy;
            if len2 <= f64::MIN_POSITIVE {
                return None;
            }
            let on = |p: [f64; 2]| -> Option<f64> {
                let t = ((p[0] - x[0]) * dx + (p[1] - x[1]) * dy) / len2;
                ((0.0..=1.0).contains(&t)
                    && (p[0] - (x[0] + dx * t)).hypot(p[1] - (x[1] + dy * t)) <= near)
                    .then_some(t)
            };
            for dir in [1isize, -1isize] {
                let mut out: Vec<usize> = Vec::new();
                let mut i = a;
                // The path has to *advance* along the chord, never double back:
                // a sphere's ring turns round at a pole, and points either side
                // of the turn lie on the chord while running back the way they
                // came.
                let mut last = 0.0f64;
                while out.len() + 2 <= n {
                    i = ((i as isize + dir).rem_euclid(n as isize)) as usize;
                    if i == b {
                        // Only when it is actually wrong. A chord along a
                        // straight rim — a cylinder's ruling — is exactly the
                        // surface, and replacing it costs a triangle per ring
                        // vertex for nothing.
                        let (pa, pb) = (
                            surface.point(ring_uv[a][0], ring_uv[a][1]),
                            surface.point(ring_uv[b][0], ring_uv[b][1]),
                        );
                        let sags = out.iter().any(|&m| {
                            let q = surface.point(ring_uv[m][0], ring_uv[m][1]);
                            let (u, w) = (v3::sub(pb, pa), v3::sub(q, pa));
                            let len2 = v3::dot(u, u);
                            let t = if len2 > f64::MIN_POSITIVE {
                                (v3::dot(w, u) / len2).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            v3::dist(q, v3::add(pa, v3::scale(u, t))) > tolerance
                        });
                        return (sags && !out.is_empty()).then_some(out);
                    }
                    let Some(t) = on(ring_uv[i]).filter(|t| *t > last) else {
                        break;
                    };
                    last = t;
                    out.push(i);
                }
            }
            None
        };

        for _ in 0..4 {
            // Decided by *edge*, and from every triangle — flat ones included.
            //
            // Per *triangle* it would replace a chord on one side and leave it
            // on the other, and the result comes back with an edge open; keying
            // by edge is what avoids that. Skipping the flat ones when *finding*
            // the chords is not the same guard, and it misses a whole class:
            // three corners on one rim is a triangle with no area in parameter
            // space, so a chord between two of them was never offered to
            // `between` and never fanned. A bored ball cut a second time at 2e-4
            // kept a 66-degree chord across its top rim, 1.09 long, with nothing
            // on the other side of it — the last open edge in that result, and
            // the reason the cut declined at that tolerance and at no other.
            let mut paths: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for t in faces.iter() {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    if let std::collections::hash_map::Entry::Vacant(slot) = paths.entry(key) {
                        if let Some(mid) = between(key.0, key.1) {
                            slot.insert(mid);
                        }
                    }
                }
            }
            if paths.is_empty() {
                break;
            }
            // Every vertex of a run being replaced. The slivers tiling such a
            // run do not name the chord — they join the run's own points to
            // each other — so they cannot be found by the edge alone.
            let mut run: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for ((a, b), mid) in &paths {
                run.insert(*a);
                run.insert(*b);
                run.extend(mid.iter().copied());
            }
            let mut next: Vec<[usize; 3]> = Vec::with_capacity(faces.len());
            for t in &faces {
                // The strip between a chord and the boundary it shortcuts is
                // about to be covered by the fan. Whatever was covering it
                // before goes, and that is the flat triangles lying wholly
                // inside the run.
                if flat(t) && t.iter().all(|v| run.contains(v)) {
                    continue;
                }
                let hit = (0..3).find_map(|k| {
                    let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                    paths.get(&(a.min(b), a.max(b))).map(|m| (a, b, c, m))
                });
                let Some((a, b, c, mid)) = hit else {
                    next.push(*t);
                    continue;
                };
                let walk: Vec<usize> = if a < b {
                    mid.clone()
                } else {
                    mid.iter().rev().copied().collect()
                };
                let mut prev = a;
                for m in walk {
                    next.push([prev, m, c]);
                    prev = m;
                }
                next.push([prev, b, c]);
            }
            faces = next;
        }
    }

    let emit = |uv: [f64; 2], positions: &mut Vec<f32>, normals: &mut Vec<f32>| -> u32 {
        let p = surface.point(uv[0], uv[1]);
        let mut nrm = surface.normal(uv[0], uv[1]).unwrap_or([0.0, 0.0, 1.0]);
        if flipped {
            nrm = v3::scale(nrm, -1.0);
        }
        positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
        normals.extend_from_slice(&[nrm[0] as f32, nrm[1] as f32, nrm[2] as f32]);
        (positions.len() / 3 - 1) as u32
    };

    // Six levels is a 4096-fold area increase; past that the tolerance is not
    // going to be met by subdivision alone and stopping beats hanging.
    for _ in 0..6 {
        let mut split: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &faces {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if boundary(a, b) {
                    continue;
                }
                let key = (a.min(b), a.max(b));
                if split.contains_key(&key) {
                    continue;
                }
                let mid = [0.5 * (uv[a][0] + uv[b][0]), 0.5 * (uv[a][1] + uv[b][1])];
                let chord = v3::scale(
                    v3::add(
                        surface.point(uv[a][0], uv[a][1]),
                        surface.point(uv[b][0], uv[b][1]),
                    ),
                    0.5,
                );
                if v3::dist(surface.point(mid[0], mid[1]), chord) <= tolerance {
                    continue;
                }
                uv.push(mid);
                let vid = emit(mid, positions, normals);
                id.push(vid);
                split.insert(key, uv.len() - 1);
            }
        }
        if split.is_empty() {
            break;
        }

        // Re-triangulate by how many of each triangle's edges were split. The
        // one- and two-edge cases exist so a refined triangle still meets its
        // unrefined neighbour edge-to-edge.
        let mut next: Vec<[usize; 3]> = Vec::with_capacity(faces.len() * 2);
        for t in &faces {
            let m: Vec<Option<usize>> = (0..3)
                .map(|k| {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    split.get(&(a.min(b), a.max(b))).copied()
                })
                .collect();
            match (m[0], m[1], m[2]) {
                (None, None, None) => next.push(*t),
                (Some(x), Some(y), Some(z)) => {
                    next.push([t[0], x, z]);
                    next.push([x, t[1], y]);
                    next.push([z, y, t[2]]);
                    next.push([x, y, z]);
                }
                (Some(x), Some(y), None) => {
                    next.push([t[0], x, t[2]]);
                    next.push([x, t[1], y]);
                    next.push([x, y, t[2]]);
                }
                (None, Some(y), Some(z)) => {
                    next.push([t[1], y, t[0]]);
                    next.push([y, t[2], z]);
                    next.push([y, z, t[0]]);
                }
                (Some(x), None, Some(z)) => {
                    next.push([t[2], z, t[1]]);
                    next.push([z, t[0], x]);
                    next.push([z, x, t[1]]);
                }
                (Some(x), None, None) => {
                    next.push([t[0], x, t[2]]);
                    next.push([x, t[1], t[2]]);
                }
                (None, Some(y), None) => {
                    next.push([t[1], y, t[0]]);
                    next.push([y, t[2], t[0]]);
                }
                (None, None, Some(z)) => {
                    next.push([t[2], z, t[1]]);
                    next.push([z, t[0], t[1]]);
                }
            }
        }
        faces = next;
    }

    faces
        .iter()
        .map(|t| [id[t[0]], id[t[1]], id[t[2]]])
        .collect()
}

/// The centre and width of a set of parameter values.
///
/// On a periodic parameter this is the *smallest arc containing them all*, not
/// the min-to-max span: values at −π and +π are the same place, and treating
/// them as a full turn apart is how a rim that sits on the branch cut gets
/// mistaken for one that circles the whole surface.
fn extent_of(values: &[f64], periodic: bool) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let lo = values.iter().cloned().fold(f64::MAX, f64::min);
    let hi = values.iter().cloned().fold(f64::MIN, f64::max);
    if !periodic {
        return (0.5 * (lo + hi), hi - lo);
    }
    use std::f64::consts::TAU;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The widest gap between neighbours is the part the values do *not* cover,
    // so the rest is the arc they do.
    let mut gap = sorted[0] + TAU - sorted[sorted.len() - 1];
    let mut after = sorted[0];
    for w in sorted.windows(2) {
        if w[1] - w[0] > gap {
            gap = w[1] - w[0];
            after = w[1];
        }
    }
    let width = TAU - gap;
    let mut mid = after + width * 0.5;
    while mid > std::f64::consts::PI {
        mid -= TAU;
    }
    (mid, width)
}

/// Give every triangle of the mesh a consistent, outward winding.
///
/// Two steps, because they fail for different reasons:
///
/// 1. **Propagate.** Walk triangle adjacency and flip any neighbour that
///    traverses a shared edge in the *same* direction — on a manifold surface
///    consistent neighbours traverse it oppositely. This needs nothing from the
///    B-rep: it is a property of the mesh. It is what fixes a face whose
///    parameterization is left-handed relative to its neighbours, which is not
///    exotic — a cylinder's side grid winds against its own caps.
///
/// 2. **Choose the sign.** Propagation makes a component consistent but cannot
///    say which way is out. Settle that by majority vote against the analytic
///    surface normals, so one sliver with an unreliable normal cannot invert a
///    whole shell, and so an intentionally-inward inner shell keeps its
///    orientation instead of being forced positive.
///
/// Closure alone never caught any of this: a mesh whose faces disagree about
/// which way is out still uses every edge exactly twice. It showed up only as a
/// volume — a radius-2, height-6 cylinder measuring 25.1 instead of 75.4.
fn orient(
    surfaces: &[Surface],
    faces: &[Face],
    triangle_face: &[u32],
    positions: &[f32],
    indices: &mut [u32],
) {
    let count = indices.len() / 3;
    if count == 0 {
        return;
    }

    // Undirected edge -> the triangles using it. Two is the manifold case; an
    // edge used more often is left alone rather than guessed at.
    let mut by_edge: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for t in 0..count {
        let tri = &indices[t * 3..t * 3 + 3];
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }

    // 1. Propagate consistency through each connected component.
    let mut component = vec![usize::MAX; count];
    let mut components = 0usize;
    let mut stack = Vec::new();
    for seed in 0..count {
        if component[seed] != usize::MAX {
            continue;
        }
        component[seed] = components;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let Some(users) = by_edge.get(&(a.min(b), a.max(b))) else {
                    continue;
                };
                if users.len() != 2 {
                    continue;
                }
                let n = if users[0] == t { users[1] } else { users[0] };
                if n == t || component[n] != usize::MAX {
                    continue;
                }
                // Same direction in both => the neighbour is wound backwards.
                let other = [indices[n * 3], indices[n * 3 + 1], indices[n * 3 + 2]];
                let same = (0..3).any(|j| other[j] == a && other[(j + 1) % 3] == b);
                if same {
                    indices.swap(n * 3 + 1, n * 3 + 2);
                }
                component[n] = components;
                stack.push(n);
            }
        }
        components += 1;
    }

    // 2. Per component, agree with the analytic normals or reverse wholesale.
    let get = |k: u32| -> V3 {
        let o = k as usize * 3;
        [
            positions[o] as f64,
            positions[o + 1] as f64,
            positions[o + 2] as f64,
        ]
    };
    let mut vote = vec![0i64; components];
    for t in 0..count {
        let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
        let (a, b, c) = (get(tri[0]), get(tri[1]), get(tri[2]));
        let n = v3::cross(v3::sub(b, a), v3::sub(c, a));
        if v3::norm(n) < 1e-18 {
            continue;
        }
        let Some(face) = triangle_face.get(t).and_then(|f| faces.get(*f as usize)) else {
            continue;
        };
        let Some(surface) = surfaces.get(face.surface) else {
            continue;
        };
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let Some(outward) = surface
            .invert(centroid)
            .and_then(|(u, v)| surface.normal(u, v))
        else {
            continue;
        };
        let outward = if face.flipped {
            v3::scale(outward, -1.0)
        } else {
            outward
        };
        vote[component[t]] += if v3::dot(n, outward) > 0.0 { 1 } else { -1 };
    }
    for t in 0..count {
        if vote[component[t]] < 0 {
            indices.swap(t * 3 + 1, t * 3 + 2);
        }
    }
}

/// Smallest step count whose chord deviation stays within `tolerance`.
pub(crate) fn adaptive_steps(
    tolerance: f64,
    mut deviation: impl FnMut(f64, f64) -> f64,
    a: f64,
    b: f64,
) -> usize {
    let mut steps = 1usize;
    while steps < 512 {
        let mut worst = 0.0f64;
        for i in 0..steps {
            let (p, q) = (
                a + (b - a) * i as f64 / steps as f64,
                a + (b - a) * (i + 1) as f64 / steps as f64,
            );
            worst = worst.max(deviation(p, q));
        }
        if worst <= tolerance {
            return steps;
        }
        steps *= 2;
    }
    steps
}

/// Put vertices along a segment until it is within `tolerance` of both surfaces.
///
/// The midpoint of a chord is pulled onto both surfaces by alternating
/// projection, so the point added lies on the curve the two share rather than on
/// the straight line between its ends.
#[allow(clippy::too_many_arguments)]
fn subdivide(
    p: V3,
    q: V3,
    a: &Surface,
    b: &Surface,
    tolerance: f64,
    depth: usize,
    vertices: &mut Vec<V3>,
    out: &mut Vec<usize>,
) {
    const MAX_DEPTH: usize = 10;
    if depth >= MAX_DEPTH {
        return;
    }
    let mid = v3::scale(v3::add(p, q), 0.5);
    if a.distance(mid).max(b.distance(mid)) <= tolerance {
        return;
    }
    // Pull the midpoint onto both surfaces by alternating projection.
    let mut m = mid;
    for _ in 0..16 {
        if a.distance(m) < 1e-12 && b.distance(m) < 1e-12 {
            break;
        }
        let Some((au, av)) = a.invert(m) else { return };
        m = a.point(au, av);
        let Some((bu, bv)) = b.invert(m) else { return };
        m = b.point(bu, bv);
    }
    if a.distance(m).max(b.distance(m)) > 1e-6 {
        return;
    }
    subdivide(p, m, a, b, tolerance, depth + 1, vertices, out);
    vertices.push(m);
    out.push(vertices.len() - 1);
    subdivide(m, q, a, b, tolerance, depth + 1, vertices, out);
}

fn centroid(v: &[V3; 3]) -> V3 {
    [
        (v[0][0] + v[1][0] + v[2][0]) / 3.0,
        (v[0][1] + v[1][1] + v[2][1]) / 3.0,
        (v[0][2] + v[1][2] + v[2][2]) / 3.0,
    ]
}

fn degenerate_at(positions: &[f32], a: u32, b: u32, c: u32) -> bool {
    let g = |k: u32| {
        let o = k as usize * 3;
        [
            positions[o] as f64,
            positions[o + 1] as f64,
            positions[o + 2] as f64,
        ]
    };
    let (pa, pb, pc) = (g(a), g(b), g(c));
    let ab = v3::sub(pb, pa);
    let ac = v3::sub(pc, pa);
    let area = v3::norm(v3::cross(ab, ac));
    let scale = v3::norm(ab).max(v3::norm(ac)).max(v3::dist(pb, pc));
    area <= 1e-12 * scale * scale
}

fn count_boundary_edges(indices: &[u32]) -> usize {
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    for t in indices.chunks_exact(3) {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::{BoxGeometry, CylinderGeometry};

    const TAU: f32 = std::f32::consts::PI * 2.0;

    #[test]
    fn a_capped_cylinder_becomes_a_body_with_three_faces_and_two_edges() {
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let body = Body::from_tagged_mesh(&g).expect("tagged");
        assert_eq!(body.faces().len(), 3, "side plus two caps");
        assert_eq!(body.edges().len(), 2, "one rim per cap");
        for e in body.edges() {
            assert!(e.closed);
            // Both faces of an edge reference the *same* vertex indices — the
            // property the whole type exists for.
            assert!(e.vertices.len() >= 4);
        }
    }

    #[test]
    fn a_capped_cylinder_tessellates_closed() {
        // The case mesh-side re-tessellation could not close: the side wants many
        // more samples around the rim than the cap has, and both must use the
        // same vertices rather than the same parameters.
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        body.refine_edges(1e-3);
        let (out, report) = body.tessellate(1e-3);

        assert_eq!(report.carried_through, 0, "a face was not filled");
        assert!(
            report.is_closed(),
            "left {} open edges",
            report.boundary_edges
        );

        // And it is the right solid: nothing beyond the radius, nothing outside
        // the height.
        for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
            let r = ((c[0] as f64).powi(2) + (c[2] as f64).powi(2)).sqrt();
            assert!(r <= 2.0 + 1e-4, "radius {r}");
            assert!((c[1] as f64).abs() <= 2.0 + 1e-4, "y {}", c[1]);
        }
    }

    #[test]
    fn a_cone_tessellates_closed() {
        let g = CylinderGeometry::new(0.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        body.refine_edges(1e-3);
        let (_, report) = body.tessellate(1e-3);
        assert_eq!(report.faces, 2);
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);
    }

    #[test]
    fn a_box_tessellates_closed() {
        let g = BoxGeometry::new(2.0, 3.0, 4.0);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        body.refine_edges(1e-3);
        let (_, report) = body.tessellate(1e-3);
        assert_eq!(report.faces, 6);
        assert_eq!(report.edges, 12);
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);
    }

    #[test]
    fn refining_edges_only_inserts_points() {
        // The originals are where two faces already agree; moving or dropping one
        // is the one way refinement could open a seam.
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 8, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        let before: Vec<Vec<V3>> = body
            .edges()
            .iter()
            .map(|e| e.vertices.iter().map(|&v| body.vertices()[v]).collect())
            .collect();

        body.refine_edges(1e-4);

        for (e, original) in body.edges().iter().zip(&before) {
            let now: Vec<V3> = e.vertices.iter().map(|&v| body.vertices()[v]).collect();
            assert!(now.len() >= original.len(), "refinement dropped points");
            for p in original {
                assert!(
                    now.iter().any(|q| v3::dist(*p, *q) < 1e-12),
                    "an original edge point was lost"
                );
            }
        }
    }

    #[test]
    fn a_finer_tolerance_costs_more_triangles_and_stays_closed() {
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 12, 1, false, 0.0, TAU);
        let mut last = 0usize;
        for &tol in &[1e-1, 1e-2, 1e-3] {
            let mut body = Body::from_tagged_mesh(&g).unwrap();
            body.refine_edges(tol);
            let (_, report) = body.tessellate(tol);
            assert!(report.is_closed(), "{tol}: {} open", report.boundary_edges);
            assert!(
                report.triangles > last,
                "{tol}: {} triangles, previous {last}",
                report.triangles
            );
            last = report.triangles;
        }
    }

    #[test]
    fn rims_of_different_refinement_still_join() {
        // The strip walk's reason for existing, asserted directly rather than
        // only through the closure of the solid.
        let g = CylinderGeometry::new(1.0, 3.0, 4.0, 20, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        body.refine_edges(1e-3);

        let counts: Vec<usize> = body.edges().iter().map(|e| e.vertices.len()).collect();
        assert_eq!(counts.len(), 2);
        assert_ne!(
            counts[0], counts[1],
            "the two rims should refine differently; if they stopped, this test \
             no longer exercises the walk"
        );

        let (_, report) = body.tessellate(1e-3);
        assert!(
            report.is_closed(),
            "{} open edges between rims of {} and {} points",
            report.boundary_edges,
            counts[0],
            counts[1]
        );
    }

    #[test]
    fn a_boundary_less_face_still_fills() {
        // A sphere and a torus are one face with *no* edges. There is nothing to
        // share, so the face samples its own footprint — and used to produce an
        // empty mesh instead, because the fill required a rim to take its
        // angular samples from.
        use crate::geometries::{SphereGeometry, TorusGeometry};

        let cases: Vec<(&str, BufferGeometry, f64)> = vec![
            ("sphere", SphereGeometry::new(2.0, 20, 12), 2.0),
            ("torus", TorusGeometry::new(3.0, 1.0, 12, 20, TAU), 4.0),
        ];
        for (name, g, scale) in cases {
            let mut body = Body::from_tagged_mesh(&g).unwrap();
            body.refine_edges(1e-3);
            let (out, report) = body.tessellate(1e-3);

            assert_eq!(report.faces, 1, "{name}");
            assert_eq!(report.edges, 0, "{name}: a closed surface has no boundary");
            assert_eq!(report.carried_through, 0, "{name}: the face was not filled");
            assert!(report.triangles > 0, "{name}: empty mesh");
            assert!(
                report.is_closed(),
                "{name}: {} open edges",
                report.boundary_edges
            );

            // And it is the right shape.
            let surface = &body.surfaces()[0];
            for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
                let p = [c[0] as f64, c[1] as f64, c[2] as f64];
                assert!(
                    surface.distance(p) < 1e-3 * scale,
                    "{name}: a vertex is {} off the surface",
                    surface.distance(p)
                );
            }
        }
    }

    #[test]
    fn an_open_cylinder_stays_open() {
        // Closure is a property of the solid, not something to force: a tube has
        // two boundaries and must come back with them.
        let g = CylinderGeometry::new(1.0, 1.0, 3.0, 16, 1, true, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        body.refine_edges(1e-3);
        let (_, report) = body.tessellate(1e-3);
        assert_eq!(report.faces, 1);
        assert_eq!(report.edges, 0);
        assert!(report.triangles > 0);
        assert!(
            !report.is_closed(),
            "an open tube should report its two open rims"
        );
    }

    #[test]
    fn every_primitive_round_trips_through_a_body() {
        use crate::geometries::{SphereGeometry, TorusGeometry};
        let cases: Vec<(&str, BufferGeometry, bool)> = vec![
            ("box", BoxGeometry::new(2.0, 3.0, 4.0), true),
            ("sphere", SphereGeometry::new(1.5, 24, 14), true),
            ("torus", TorusGeometry::new(3.0, 1.0, 12, 24, TAU), true),
            (
                "capped cylinder",
                CylinderGeometry::new(2.0, 2.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            (
                "cone",
                CylinderGeometry::new(0.0, 2.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            // A truncated cone is the case that forced the strip walk: its
            // radius-1 rim converges at 81 points where its radius-3 rim needs
            // 161, and a tensor grid cannot join rows of different widths.
            (
                "truncated cone",
                CylinderGeometry::new(1.0, 3.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            (
                "open cylinder",
                CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, true, 0.0, TAU),
                false,
            ),
        ];
        for (name, g, should_close) in cases {
            let mut body = Body::from_tagged_mesh(&g).unwrap_or_else(|| panic!("{name}: no body"));
            body.refine_edges(1e-3);
            let (_, report) = body.tessellate(1e-3);
            assert_eq!(report.carried_through, 0, "{name}: a face was not filled");
            assert!(report.triangles > 0, "{name}: empty");
            assert_eq!(
                report.is_closed(),
                should_close,
                "{name}: closed = {}, expected {should_close} ({} open edges)",
                report.is_closed(),
                report.boundary_edges
            );
        }
    }

    #[test]
    fn every_primitive_is_a_valid_solid() {
        use crate::geometries::{SphereGeometry, TorusGeometry};
        let cases: Vec<(&str, BufferGeometry, bool)> = vec![
            ("box", BoxGeometry::new(2.0, 3.0, 4.0), true),
            ("sphere", SphereGeometry::new(1.5, 24, 14), true),
            ("torus", TorusGeometry::new(3.0, 1.0, 12, 24, TAU), true),
            (
                "capped cylinder",
                CylinderGeometry::new(2.0, 2.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            (
                "cone",
                CylinderGeometry::new(0.0, 2.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            (
                "truncated cone",
                CylinderGeometry::new(1.0, 3.0, 4.0, 20, 1, false, 0.0, TAU),
                true,
            ),
            (
                "open cylinder",
                CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, true, 0.0, TAU),
                false,
            ),
        ];
        for (name, g, solid) in cases {
            let body = Body::from_tagged_mesh(&g).unwrap();
            let defects = body.defects(1e-4);
            assert_eq!(
                body.is_valid_solid(1e-4),
                solid,
                "{name}: defects {defects:?}"
            );
            // Whatever else, no reference may dangle and no edge may claim
            // surfaces it does not lie on.
            for d in &defects {
                assert!(matches!(d, Defect::EdgeFaceCount { .. }), "{name}: {d:?}");
            }
        }
    }

    #[test]
    fn a_face_may_meet_itself_along_a_seam() {
        // The manifold condition is that a curve is walked twice by the
        // boundaries around it — *not* that two distinct faces name it. A seam
        // is the case where both walks belong to one face: a cylinder's side
        // meets itself along the line where its parameterisation closes. STEP
        // writes it that way, two `ORIENTED_EDGE`s over one `EDGE_CURVE` in a
        // single face's loop.
        //
        // Nothing this crate builds uses an edge twice yet, so this is the only
        // thing holding the distinction down. Without it, counting distinct
        // faces reads the same on every existing body and the difference goes
        // unnoticed until a seam needs to become an edge.
        let mut body = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        assert!(body.is_valid_solid(1e-4), "{:?}", body.defects(1e-4));

        // The side is the face on the cylindrical surface.
        let side = body
            .faces
            .iter()
            .position(|f| matches!(body.surfaces[f.surface], Surface::Cylinder { .. }))
            .expect("a cylinder has a side");
        let surface = body.surfaces[body.faces[side].surface].clone();

        // The seam runs between the two rims at one parameter. Take a vertex of
        // the first rim and the vertex of the second that sits at the same `u`,
        // so the edge really does lie along the surface rather than across it.
        let rims = body.faces[side].edges.clone();
        assert_eq!(rims.len(), 2, "a capped cylinder's side has two rims");
        let start = body.edges[rims[0]].vertices[0];
        let u0 = surface.invert(body.vertices[start]).expect("on the side").0;
        let end = *body.edges[rims[1]]
            .vertices
            .iter()
            .min_by(|&&a, &&b| {
                let d = |v: usize| {
                    let u = surface.invert(body.vertices[v]).map_or(f64::MAX, |p| p.0);
                    (u - u0)
                        .abs()
                        .min((u - u0).abs() - std::f64::consts::TAU)
                        .abs()
                };
                d(a).total_cmp(&d(b))
            })
            .expect("the far rim has vertices");

        let seam = body.edges.len();
        body.edges.push(Edge {
            surfaces: (body.faces[side].surface, body.faces[side].surface),
            vertices: vec![start, end],
            closed: false,
        });
        // Named twice by the one face, which is what makes it a seam.
        body.faces[side].edges.push(seam);
        body.faces[side].edges.push(seam);

        let defects = body.defects(1e-4);
        assert!(
            defects.is_empty(),
            "a seam edge is not a defect: {defects:?}"
        );
        assert!(
            body.shells().iter().all(|s| s.closed),
            "adding a seam does not open a shell"
        );

        // And naming it once is still an error, for the same reason it always
        // was: one walk means an open boundary.
        body.faces[side].edges.pop();
        assert_eq!(
            body.defects(1e-4),
            vec![Defect::EdgeFaceCount {
                edge: seam,
                uses: 1
            }]
        );
    }

    #[test]
    fn euler_characteristic_distinguishes_a_sphere_from_a_torus() {
        // The genus check. A closed orientable shell of genus g has V − E + F =
        // 2 − 2g, so a box and a sphere give 2 and a torus gives 0. Getting this
        // right means the *topology* is right, not merely that the mesh closed.
        use crate::geometries::{SphereGeometry, TorusGeometry};

        let box_body = Body::from_tagged_mesh(&BoxGeometry::new(2.0, 2.0, 2.0)).unwrap();
        let shells = box_body.shells();
        assert_eq!(shells.len(), 1);
        assert!(shells[0].closed);
        assert!(
            shells[0].euler_meaningful,
            "a box has no periodic face, so nothing is unrepresented"
        );
        assert_eq!(shells[0].euler, 2, "a box is genus 0");

        // A sphere and a torus are each a single face with no edges: V = E = 0,
        // F = 1, so the count is 1 either way and cannot tell them apart. That is
        // a real limit of a body with no seam edges, not a miscount — recorded
        // so the number is not read as more than it is.
        for (name, g) in [
            ("sphere", SphereGeometry::new(1.0, 12, 8)),
            ("torus", TorusGeometry::new(3.0, 1.0, 10, 16, TAU)),
        ] {
            let b = Body::from_tagged_mesh(&g).unwrap();
            let sh = &b.shells()[0];
            assert_eq!(sh.euler, 1, "{name}");
            assert!(
                !sh.euler_meaningful,
                "{name}: its seam is unrepresented, so 1 is not a genus claim"
            );
            assert!(sh.closed, "{name}: closed by its own periodicity");
        }
    }

    #[test]
    fn a_capped_cylinder_is_one_closed_shell() {
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 16, 1, false, 0.0, TAU);
        let body = Body::from_tagged_mesh(&g).unwrap();
        let shells = body.shells();
        assert_eq!(shells.len(), 1, "side and caps are all connected");
        assert_eq!(shells[0].faces.len(), 3);
        assert!(shells[0].closed);
        // Its Euler number is *not* 2, and that is honest rather than wrong: the
        // side face is periodic, so a full B-rep would give it a seam edge and
        // two rim vertices that this body does not represent. The count is over
        // the elements that exist, and `euler_meaningful` says so.
        assert!(
            !shells[0].euler_meaningful,
            "a periodic face has an unrepresented seam"
        );
    }

    #[test]
    fn an_open_tube_is_reported_as_open_not_as_valid() {
        let g = CylinderGeometry::new(1.0, 1.0, 2.0, 12, 1, true, 0.0, TAU);
        let body = Body::from_tagged_mesh(&g).unwrap();
        assert!(!body.is_valid_solid(1e-4));
    }

    #[test]
    fn a_dangling_reference_is_a_defect_not_a_panic() {
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 12, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        let n = body.vertices().len();
        body.edges[0].vertices.push(n + 99);
        let defects = body.defects(1e-4);
        assert!(
            defects
                .iter()
                .any(|d| matches!(d, Defect::EdgeVertexMissing { .. })),
            "{defects:?}"
        );
    }

    #[test]
    fn an_edge_that_is_not_on_its_surfaces_is_caught() {
        // An edge is a *claim* that two surfaces meet along it. Measuring that
        // claim is what stops a plausible-looking body from being wrong.
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 12, 1, false, 0.0, TAU);
        let mut body = Body::from_tagged_mesh(&g).unwrap();
        let vi = body.edges[0].vertices[0];
        body.vertices[vi] = [99.0, 99.0, 99.0];
        assert!(body
            .defects(1e-4)
            .iter()
            .any(|d| matches!(d, Defect::EdgeOffSurface { .. })));
    }

    #[test]
    fn authored_bodies_are_valid_solids_and_tessellate_closed() {
        // Built topology-first, with no mesh anywhere in the input — the
        // direction a B-rep is supposed to run in.
        let cases: Vec<(&str, Body)> = vec![
            ("cuboid", Body::cuboid([2.0, 3.0, 4.0])),
            ("sphere", Body::sphere([0.0; 3], 2.0)),
            ("torus", Body::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0)),
            (
                "cylinder",
                Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0),
            ),
            ("cone", Body::cone([0.0; 3], [0.0, 0.0, 1.0], 2.0, 4.0)),
            (
                "frustum",
                Body::frustum([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0, 4.0),
            ),
        ];
        for (name, mut body) in cases {
            assert!(
                body.is_valid_solid(1e-6),
                "{name}: {:?}",
                body.defects(1e-6)
            );
            body.refine_edges(1e-3);
            let (out, report) = body.tessellate(1e-3);
            assert_eq!(report.carried_through, 0, "{name}: a face was not filled");
            assert!(report.triangles > 0, "{name}: empty");
            assert!(
                report.is_closed(),
                "{name}: {} open edges",
                report.boundary_edges
            );
            assert!(out.get_attribute("position").unwrap().count() > 0);
        }
    }

    #[test]
    fn an_authored_cuboid_has_the_topology_a_box_has() {
        let body = Body::cuboid([2.0, 2.0, 2.0]);
        assert_eq!(body.faces().len(), 6);
        assert_eq!(body.edges().len(), 12);
        assert_eq!(body.vertices().len(), 8);

        let shells = body.shells();
        assert_eq!(shells.len(), 1);
        assert!(shells[0].closed);
        assert!(
            shells[0].euler_meaningful,
            "no periodic face, nothing hidden"
        );
        assert_eq!(shells[0].euler, 2, "8 − 12 + 6");
    }

    #[test]
    fn a_transformed_body_is_still_a_solid() {
        use crate::math::{Matrix4, Quaternion, Vector3};
        let body = Body::cuboid([2.0, 4.0, 6.0]);
        let m = Matrix4::compose(
            Vector3::new(10.0, -3.0, 2.0),
            Quaternion::from_axis_angle(Vector3::new(1.0, 1.0, 0.0).normalize(), 0.7),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let moved = body
            .transform(&m)
            .expect("a rigid motion carries every surface");

        // The topology is untouched — same faces, same shared edges — so what
        // moved is still closed. A transform that only moved the vertices would
        // leave the surfaces behind and every face would be off its own plane.
        assert_eq!(moved.faces().len(), body.faces().len());
        assert_eq!(moved.edges().len(), body.edges().len());
        assert!(moved.is_valid_solid(1e-6), "{:?}", moved.defects(1e-6));

        let (_, report) = {
            let mut c = moved.clone();
            c.refine_edges(1e-4);
            c.tessellate(1e-4)
        };
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);
    }

    #[test]
    fn a_transform_that_cannot_be_represented_declines() {
        use crate::math::{Matrix4, Vector3};
        // A non-uniform scale turns a sphere into an ellipsoid, which is not one
        // of these surfaces. Returning a plausible sphere would put the geometry
        // somewhere the caller did not ask for.
        let squash = Matrix4::scale(Vector3::new(1.0, 1.0, 0.5));
        assert!(Body::sphere([0.0; 3], 1.0).transform(&squash).is_none());
        // A box is planes, and a plane survives any affine map.
        assert!(Body::cuboid([1.0, 1.0, 1.0]).transform(&squash).is_some());
    }

    #[test]
    fn translating_preserves_volume_and_moves_the_centre() {
        let a = Body::cuboid([2.0, 2.0, 2.0]);
        let b = a.translated([5.0, 0.0, 0.0]).unwrap();
        let bounds = |body: &Body| {
            let mut lo = f64::MAX;
            let mut hi = f64::MIN;
            for v in body.vertices() {
                lo = lo.min(v[0]);
                hi = hi.max(v[0]);
            }
            (lo, hi)
        };
        assert_eq!(bounds(&a), (-1.0, 1.0));
        assert_eq!(bounds(&b), (4.0, 6.0));
    }

    #[test]
    fn an_authored_body_matches_the_shape_it_names() {
        // The tessellation has to be the solid, not merely closed.
        let mut body = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0);
        body.refine_edges(1e-3);
        let (out, _) = body.tessellate(1e-3);
        for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
            let r = ((c[0] as f64).powi(2) + (c[1] as f64).powi(2)).sqrt();
            assert!(r <= 2.0 + 1e-3, "radius {r}");
            assert!((-1e-3..=5.0 + 1e-3).contains(&(c[2] as f64)), "z {}", c[2]);
        }

        let mut sphere = Body::sphere([1.0, -2.0, 0.5], 3.0);
        sphere.refine_edges(1e-3);
        let (out, _) = sphere.tessellate(1e-3);
        for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
            let d = v3::dist([c[0] as f64, c[1] as f64, c[2] as f64], [1.0, -2.0, 0.5]);
            assert!((d - 3.0).abs() < 1e-3, "radius {d}");
        }
    }

    #[test]
    fn an_authored_cone_has_one_cap_not_two() {
        // The apex is a degeneracy, not an edge: nothing meets the surface
        // there, so there is nothing to share and no face to cap it with.
        let body = Body::cone([0.0; 3], [0.0, 0.0, 1.0], 2.0, 4.0);
        assert_eq!(body.faces().len(), 2, "side and one disk");
        assert_eq!(body.edges().len(), 1, "one rim");
        assert!(body.is_valid_solid(1e-6));
    }

    #[test]
    fn a_face_with_a_hole_is_trimmed_rather_than_filled_solid() {
        // The smallest solid that needs trim loops. Its top is one plane bounded
        // by an outer rectangle *and* an inner circle — no parameter rectangle
        // describes that, and without hole bridging the plate comes back with
        // the bore filled in.
        let mut body = Body::plate_with_hole([10.0, 8.0, 2.0], 2.0).expect("fits");
        assert_eq!(body.faces().len(), 7, "six plates and a bore");
        assert_eq!(body.edges().len(), 14, "twelve box edges and two rims");
        assert!(body.is_valid_solid(1e-6), "{:?}", body.defects(1e-6));

        // The top face really does have two loops, one of them a hole.
        let top = body
            .faces()
            .iter()
            .position(|f| {
                matches!(&body.surfaces()[f.surface], Surface::Plane { normal, .. }
                    if (normal[2] - 1.0).abs() < 1e-9)
            })
            .expect("a +Z face");
        let loops = body.face_loops(top);
        assert_eq!(loops.len(), 2, "outer rectangle plus the bore");
        assert_eq!(loops.iter().filter(|l| l.is_hole()).count(), 1);

        body.refine_edges(1e-3);
        let (out, report) = body.tessellate(1e-3);
        assert_eq!(report.carried_through, 0, "a face was not filled");
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);

        // Nothing inside the bore: a solid plate would put triangles there.
        let pos = &out.get_attribute("position").unwrap().array;
        let idx = out.index.as_ref().unwrap();
        for t in idx.chunks_exact(3) {
            let mut c = [0.0f64; 3];
            for &i in t {
                let o = i as usize * 3;
                c[0] += pos[o] as f64 / 3.0;
                c[1] += pos[o + 1] as f64 / 3.0;
                c[2] += pos[o + 2] as f64 / 3.0;
            }
            let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
            let on_a_cap = (c[2].abs() - 1.0).abs() < 1e-6;
            assert!(
                !(on_a_cap && r < 2.0 - 1e-3),
                "a cap triangle sits inside the bore at radius {r}"
            );
        }
    }

    #[test]
    fn a_hole_that_does_not_fit_is_refused() {
        assert!(Body::plate_with_hole([4.0, 4.0, 1.0], 2.0).is_none());
        assert!(Body::plate_with_hole([4.0, 4.0, 1.0], 0.0).is_none());
        assert!(Body::plate_with_hole([4.0, 4.0, 1.0], 1.9).is_some());
    }

    #[test]
    fn trim_loops_are_oriented_so_a_hole_reads_as_one() {
        // Whichever way the input wound, the outer loop comes back positive and
        // the holes negative — the fill relies on it.
        let body = Body::plate_with_hole([10.0, 8.0, 2.0], 2.0).unwrap();
        for fi in 0..body.faces().len() {
            // A periodic face's "loops" are straight lines at constant `v`, with
            // zero signed area — outer and hole are not meaningful there.
            let f = &body.faces()[fi];
            if f.u_wraps || f.v_wraps {
                continue;
            }
            let loops = body.face_loops(fi);
            if loops.is_empty() {
                continue;
            }
            assert_eq!(
                loops.iter().filter(|l| l.is_outer()).count(),
                1,
                "face {fi} has more than one outer loop"
            );
            let outer = loops.iter().find(|l| l.is_outer()).unwrap();
            for h in loops.iter().filter(|l| l.is_hole()) {
                assert!(
                    h.area.abs() < outer.area.abs(),
                    "face {fi}: a hole is larger than its outer loop"
                );
            }
        }
    }

    #[test]
    fn without_provenance_there_is_no_body() {
        let mut g = CylinderGeometry::new(1.0, 1.0, 2.0, 8, 1, false, 0.0, TAU);
        g.surfaces = None;
        assert!(Body::from_tagged_mesh(&g).is_none());
    }
}

/// Join two rows of a swept face, which need not have the same number of points.
///
/// # Why not a tensor grid
///
/// Refinement is driven by curvature, so the two rims of a truncated cone
/// converge at different counts — 81 points at radius 1 against 161 at radius 3.
/// Both are correct on their own, and a tensor grid cannot connect them: it
/// needs its rows the same width, so the wider rim ends up unattached to its cap
/// and the body is open along it.
///
/// Imposing one sampling on both rims was tried and is worse — it fights the
/// tolerance that produced them, and made a working capped cylinder fail. The
/// answer is to stop requiring equal widths: walk the two rows together,
/// advancing whichever side is *behind in parameter*, and emit a triangle for
/// each advance. Every point on both rows is used, in order, and the strip is
/// closed by construction.
#[allow(clippy::too_many_arguments)]
fn strip(
    lower: &[u32],
    lower_u: &[f64],
    upper: &[u32],
    upper_u: &[f64],
    wraps: bool,
    flipped: bool,
    positions: &[f32],
    indices: &mut Vec<u32>,
) {
    if lower.len() < 2 || upper.len() < 2 {
        return;
    }
    let (nl, nu) = (lower.len(), upper.len());
    // How far to walk: a closed sweep returns to its start, an open one stops.
    let (steps_l, steps_u) = if wraps { (nl, nu) } else { (nl - 1, nu - 1) };

    let (mut i, mut j) = (0usize, 0usize);
    let emit = |a: u32, b: u32, c: u32, indices: &mut Vec<u32>| {
        if degenerate_at(positions, a, b, c) {
            return;
        }
        if flipped {
            indices.extend_from_slice(&[a, c, b]);
        } else {
            indices.extend_from_slice(&[a, b, c]);
        }
    };

    while i < steps_l || j < steps_u {
        // Fractional progress along each row; advance the one that is behind, so
        // the triangles stay well shaped rather than fanning from one corner.
        let pl = i as f64 / steps_l as f64;
        let pu = j as f64 / steps_u as f64;
        let take_lower = if i >= steps_l {
            false
        } else if j >= steps_u {
            true
        } else {
            pl <= pu
        };

        if take_lower {
            let a = lower[i % nl];
            let b = lower[(i + 1) % nl];
            let c = upper[j % nu];
            emit(a, b, c, indices);
            i += 1;
        } else {
            let a = upper[j % nu];
            let b = lower[i % nl];
            let c = upper[(j + 1) % nu];
            emit(a, c, b, indices);
            j += 1;
        }
    }
    let _ = (lower_u, upper_u);
}

/// What is wrong with a body, if anything.
///
/// Reported rather than asserted: a body assembled from a mesh, or later from a
/// boolean, can be malformed in ways that are cheap to detect and expensive to
/// discover downstream. Each variant names something specific enough to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Defect {
    /// An edge used by other than two face boundaries. One means an open
    /// boundary; three or more means a non-manifold junction.
    ///
    /// *Uses*, not distinct faces: the manifold condition is that a curve is
    /// walked twice by the boundaries around it, and both walks can belong to
    /// the same face. A seam is exactly that — a cylinder's face meets itself
    /// along it — and STEP says so directly, with two `ORIENTED_EDGE`s over one
    /// `EDGE_CURVE` in a single face's loop. Counting distinct faces gives the
    /// same answer whenever no face uses an edge twice, which is every body this
    /// crate builds today, and the wrong answer for the first one that does.
    EdgeFaceCount { edge: usize, uses: usize },
    /// A face naming a surface that does not exist.
    FaceSurfaceMissing { face: usize, surface: usize },
    /// An edge naming a vertex that does not exist.
    EdgeVertexMissing { edge: usize, vertex: usize },
    /// A closed edge whose first and last vertices differ.
    EdgeNotClosed { edge: usize },
    /// An edge whose vertices are not on both the surfaces it claims to join.
    EdgeOffSurface { edge: usize, deviation_scaled: u64 },
}

/// A connected group of faces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shell {
    pub faces: Vec<usize>,
    /// Every edge is used by two of this shell's faces, *and* no face has a free
    /// end.
    ///
    /// The second half is not redundant. An open tube is a single face with
    /// **no edges at all** — its two rims border nothing, so there is nothing to
    /// count — and a check that only looked at edge use would call it closed
    /// because it has none.
    pub closed: bool,
    /// `V − E + F` over the elements this body actually represents.
    ///
    /// Equal to `2 − 2g` **only when [`Self::euler_meaningful`]**. A periodic
    /// face has a seam, and a revolved one may have a pole; a full B-rep gives
    /// those seam edges and pole vertices, and this body does not. Without them
    /// the alternating sum is simply counting different things — a capped
    /// cylinder comes to 3 rather than 2, because the side's seam edge and its
    /// two rim vertices are missing.
    pub euler: i64,
    /// Is every face's boundary represented by edges — no seam, no pole?
    pub euler_meaningful: bool,
}

impl Body {
    /// Group faces into connected shells.
    ///
    /// A body is not necessarily one solid — a boolean can leave several, and a
    /// solid with a cavity has an outer shell and an inner one. Faces are
    /// connected when they share an edge.
    pub fn shells(&self) -> Vec<Shell> {
        let mut owner: Vec<Option<usize>> = vec![None; self.faces.len()];
        let mut shells: Vec<Shell> = Vec::new();

        for seed in 0..self.faces.len() {
            if owner[seed].is_some() {
                continue;
            }
            let id = shells.len();
            let mut group = vec![seed];
            owner[seed] = Some(id);
            let mut queue = vec![seed];
            while let Some(f) = queue.pop() {
                for &e in &self.faces[f].edges {
                    for (g, other) in self.faces.iter().enumerate() {
                        if owner[g].is_none() && other.edges.contains(&e) {
                            owner[g] = Some(id);
                            group.push(g);
                            queue.push(g);
                        }
                    }
                }
            }

            // Elements belonging to this shell.
            let mut edges: Vec<usize> = group
                .iter()
                .flat_map(|&f| self.faces[f].edges.iter().copied())
                .collect();
            edges.sort_unstable();
            edges.dedup();

            // Uses, not distinct faces — see `Defect::EdgeFaceCount`.
            let edges_shared = edges.iter().all(|&e| {
                group
                    .iter()
                    .map(|&f| self.faces[f].edges.iter().filter(|&&x| x == e).count())
                    .sum::<usize>()
                    == 2
            });
            let no_free_ends = group.iter().all(|&f| self.face_free_ends(f) == 0);
            let closed = edges_shared && no_free_ends;
            let euler_meaningful = group.iter().all(|&f| {
                let face = &self.faces[f];
                !face.u_wraps && !face.v_wraps && self.face_degenerate_ends(f) == 0
            });

            // Vertices, and edge *segments* — an edge here is a polyline, and
            // Euler counts topological edges, so a closed loop of `n` points
            // contributes `n − 1` segments and `n − 1` vertices.
            let mut vertices: Vec<usize> = Vec::new();
            let mut segments = 0i64;
            for &e in &edges {
                let edge = &self.edges[e];
                let n = edge.vertices.len();
                if n < 2 {
                    continue;
                }
                segments += (n - 1) as i64;
                let last = if edge.closed { n - 1 } else { n };
                vertices.extend(&edge.vertices[..last]);
            }
            vertices.sort_unstable();
            vertices.dedup();

            shells.push(Shell {
                euler: vertices.len() as i64 - segments + group.len() as i64,
                faces: group,
                closed,
                euler_meaningful,
            });
        }
        shells
    }

    /// Ends of a face's parameter footprint that nothing closes: no shared
    /// edge, no degeneracy, no wrap.
    ///
    /// These are *free boundaries* — an open tube's two rims. They are the
    /// reason closure cannot be read off edge counts alone: a face with no edges
    /// may be a sphere (closed by its own periodicity) or a tube (not closed at
    /// all), and only its footprint distinguishes them.
    pub fn face_free_ends(&self, face_index: usize) -> usize {
        let face = &self.faces[face_index];

        // A free end is a vertex an odd number of the face's segments meet at.
        //
        // This used to ask, for a plane, whether the edges *chain* into a loop —
        // walk greedily from the first segment and take whatever can be reached
        // — and for anything else where its boundary sat in the parameters. The
        // first answers a harder question badly: a spur, a second ring, or three
        // segments at a point all stop the walk and it cannot say which. The
        // second answers a different question altogether, and got this one
        // wrong: a sphere with a square column bored through it has every vertex
        // of its boundary at even degree and still read as having two free ends.
        //
        // Degree is the definition. Every vertex of a closed boundary is an end
        // of an even number of its segments; an open polyline has exactly two
        // that are not. It does not care how many rings there are, what order
        // they come in, where they touch, or what surface they are on.
        let mut degree: HashMap<usize, usize> = HashMap::new();
        for &e in &face.edges {
            for w in self.edges[e].vertices.windows(2) {
                if w[0] == w[1] {
                    continue;
                }
                *degree.entry(w[0]).or_insert(0) += 1;
                *degree.entry(w[1]).or_insert(0) += 1;
            }
        }
        // No edges at all means one of two opposite things, and the surface
        // says which. A sphere or a torus is closed on its own — a whole ball is
        // one face bounded by nothing, and it is a solid. A *plane* bounded by
        // nothing is not a boundary at all, and neither is an open tube, whose
        // two rims border nothing and which a check on edge use alone would call
        // closed because it has none.
        if degree.is_empty() {
            // Closed on its own, or not closed at all. A ball is one face
            // bounded by nothing and it is a solid: its `u` goes right round and
            // its `v` ends are *poles*, single points with no length to them. A
            // torus is the same with both wrapping. An open tube also has no
            // edges and is not closed — its `v` ends are circles, and they
            // border nothing.
            let wraps_v = face.v_wraps || self.face_degenerate_ends(face_index) == 2;
            return if face.u_wraps && wraps_v { 0 } else { 2 };
        }
        degree.values().filter(|d| *d % 2 == 1).count()
    }

    /// Ends where the surface itself degenerates — a sphere's pole, a cone's
    /// apex. Those close the face without needing an edge.
    pub(crate) fn face_degenerate_ends(&self, face_index: usize) -> usize {
        let face = &self.faces[face_index];
        let surface = &self.surfaces[face.surface];
        [face.v_range.0, face.v_range.1]
            .iter()
            .filter(|&&end| self.end_is_degenerate(face, surface, end))
            .count()
    }

    fn end_is_degenerate(&self, face: &Face, surface: &Surface, v: f64) -> bool {
        let (u0, u1) = face.u_range;
        let a = surface.point(u0, v);
        let b = surface.point(0.5 * (u0 + u1), v);
        let scale = (u1 - u0).abs().max(1.0);
        v3::dist(a, b) < scale * 1e-9
    }

    /// Everything structurally wrong with this body.
    ///
    /// Empty means the topology is sound: every edge joins exactly two faces,
    /// every reference resolves, and every edge really does lie on both the
    /// surfaces it claims to join. That last one is the geometric check — an
    /// edge is a *claim* that two surfaces meet along it, and measuring it is
    /// what stops a plausible-looking body from being wrong.
    pub fn defects(&self, tolerance: f64) -> Vec<Defect> {
        let mut out = Vec::new();

        for (fi, face) in self.faces.iter().enumerate() {
            if face.surface >= self.surfaces.len() {
                out.push(Defect::FaceSurfaceMissing {
                    face: fi,
                    surface: face.surface,
                });
            }
        }

        for (ei, edge) in self.edges.iter().enumerate() {
            for &v in &edge.vertices {
                if v >= self.vertices.len() {
                    out.push(Defect::EdgeVertexMissing {
                        edge: ei,
                        vertex: v,
                    });
                }
            }
            if edge.closed && edge.vertices.first() != edge.vertices.last() {
                out.push(Defect::EdgeNotClosed { edge: ei });
            }

            let uses: usize = self
                .faces
                .iter()
                .map(|f| f.edges.iter().filter(|&&e| e == ei).count())
                .sum();
            if uses != 2 {
                out.push(Defect::EdgeFaceCount { edge: ei, uses });
            }

            let (a, b) = (edge.surfaces.0, edge.surfaces.1);
            if a < self.surfaces.len() && b < self.surfaces.len() {
                let mut worst = 0.0f64;
                for &v in &edge.vertices {
                    if v < self.vertices.len() {
                        let p = self.vertices[v];
                        worst = worst
                            .max(self.surfaces[a].distance(p))
                            .max(self.surfaces[b].distance(p));
                    }
                }
                if worst > tolerance {
                    out.push(Defect::EdgeOffSurface {
                        edge: ei,
                        // Scaled to an integer so the variant stays comparable;
                        // a float there would make two defects "different" over
                        // a rounding difference.
                        deviation_scaled: (worst / tolerance) as u64,
                    });
                }
            }
        }
        out
    }

    /// Is the topology sound and every shell closed?
    pub fn is_valid_solid(&self, tolerance: f64) -> bool {
        self.defects(tolerance).is_empty() && self.shells().iter().all(|s| s.closed)
    }
}

/// Authoring: build a body directly from surfaces and edges.
///
/// Until now a `Body` could only be *derived* from a tagged mesh, which meant it
/// could never represent anything the mesh path had not already produced — a
/// view, not a model. These constructors build the topology first and let the
/// mesh come out of it, which is the direction a B-rep is supposed to run in.
///
/// The edges are laid out at a nominal resolution and then refined by
/// [`Body::refine_edges`]; they are shared vertex lists from the start, so a
/// body assembled here is watertight for the same reason one recovered from a
/// mesh is.
impl Body {
    /// The same solid, moved.
    ///
    /// `None` when a surface cannot be carried through the transform — a
    /// non-uniform scale turns a sphere into an ellipsoid, which is not one of
    /// these surfaces, and returning a *plausible* sphere there would put the
    /// geometry somewhere the caller did not ask for. The topology is untouched:
    /// the same faces, the same shared edges, so a transformed solid is still a
    /// solid.
    pub fn transform(&self, m: &Matrix4) -> Option<Body> {
        let surfaces: Option<Vec<Surface>> = self.surfaces.iter().map(|s| s.transform(m)).collect();
        // Moved in f64 through the same affine the surfaces use, rather than
        // through `Matrix4`'s f32: a vertex that lands even slightly off its
        // surface stops being shared, and the seam it was holding closed opens.
        let a = super::surface::Affine::from_matrix4(m)?;
        let vertices = self.vertices.iter().map(|p| a.point(*p)).collect();
        Some(Body {
            surfaces: surfaces?,
            vertices,
            edges: self.edges.clone(),
            faces: self.faces.clone(),
        })
    }

    /// The same solid, translated.
    pub fn translated(&self, offset: V3) -> Option<Body> {
        self.transform(&Matrix4::translation(crate::math::Vector3::new(
            offset[0] as f32,
            offset[1] as f32,
            offset[2] as f32,
        )))
    }

    /// An axis-aligned box centred on the origin: six planar faces, twelve
    /// edges, eight vertices.
    ///
    /// The only primitive here whose boundary is entirely represented — so the
    /// only one whose Euler number is a genus claim.
    pub fn cuboid(size: V3) -> Body {
        let (hx, hy, hz) = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5);
        let corner = |i: usize| -> V3 {
            [
                if i & 1 == 0 { -hx } else { hx },
                if i & 2 == 0 { -hy } else { hy },
                if i & 4 == 0 { -hz } else { hz },
            ]
        };
        let vertices: Vec<V3> = (0..8).map(corner).collect();

        let surfaces = vec![
            Surface::plane([hx, 0.0, 0.0], [1.0, 0.0, 0.0]),
            Surface::plane([-hx, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            Surface::plane([0.0, hy, 0.0], [0.0, 1.0, 0.0]),
            Surface::plane([0.0, -hy, 0.0], [0.0, -1.0, 0.0]),
            Surface::plane([0.0, 0.0, hz], [0.0, 0.0, 1.0]),
            Surface::plane([0.0, 0.0, -hz], [0.0, 0.0, -1.0]),
        ];

        // The twelve edges, each named by the two faces that meet along it and
        // the two corners it runs between.
        let spec: [(usize, usize, usize, usize); 12] = [
            (0, 2, 3, 7),
            (0, 3, 1, 5),
            (0, 4, 5, 7),
            (0, 5, 1, 3),
            (1, 2, 2, 6),
            (1, 3, 0, 4),
            (1, 4, 4, 6),
            (1, 5, 0, 2),
            (2, 4, 6, 7),
            (2, 5, 2, 3),
            (3, 4, 4, 5),
            (3, 5, 0, 1),
        ];
        let edges: Vec<Edge> = spec
            .iter()
            .map(|&(a, b, v0, v1)| Edge {
                surfaces: (a.min(b), a.max(b)),
                vertices: vec![v0, v1],
                closed: false,
            })
            .collect();

        let faces = (0..6)
            .map(|si| Face {
                surface: si,
                edges: (0..edges.len())
                    .filter(|&e| edges[e].surfaces.0 == si || edges[e].surfaces.1 == si)
                    .collect(),
                flipped: false,
                u_range: (-hx.max(hy).max(hz) * 2.0, hx.max(hy).max(hz) * 2.0),
                v_range: (-hx.max(hy).max(hz) * 2.0, hx.max(hy).max(hz) * 2.0),
                u_wraps: false,
                v_wraps: false,
                loops: None,
            })
            .collect();

        Body {
            surfaces,
            vertices,
            edges,
            faces,
        }
    }

    /// A closed sphere: one face, no edges. Closed by its own periodicity.
    pub fn sphere(center: V3, radius: f64) -> Body {
        Body::single_face(
            Surface::sphere(center, radius),
            (-std::f64::consts::PI, std::f64::consts::PI),
            (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            true,
            false,
        )
    }

    /// A closed torus: one face, no edges, periodic in both directions.
    pub fn torus(center: V3, axis: V3, major: f64, minor: f64) -> Body {
        Body::single_face(
            Surface::torus(center, axis, major, minor),
            (-std::f64::consts::PI, std::f64::consts::PI),
            (-std::f64::consts::PI, std::f64::consts::PI),
            true,
            true,
        )
    }

    /// A capped cylinder: side plus two disks, joined at two rim edges.
    pub fn cylinder(base: V3, axis: V3, radius: f64, height: f64) -> Body {
        Body::revolved(base, axis, radius, radius, height)
    }

    /// A cone: side plus one disk. The apex is a degeneracy, not an edge —
    /// nothing meets the surface there, so there is nothing to share.
    pub fn cone(base: V3, axis: V3, radius: f64, height: f64) -> Body {
        Body::revolved(base, axis, 0.0, radius, height)
    }

    /// A truncated cone, or a cylinder when the radii match.
    pub fn frustum(base: V3, axis: V3, radius_bottom: f64, radius_top: f64, height: f64) -> Body {
        Body::revolved(base, axis, radius_top, radius_bottom, height)
    }

    fn single_face(
        surface: Surface,
        u_range: (f64, f64),
        v_range: (f64, f64),
        u_wraps: bool,
        v_wraps: bool,
    ) -> Body {
        Body {
            surfaces: vec![surface],
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: vec![Face {
                surface: 0,
                edges: Vec::new(),
                flipped: false,
                u_range,
                v_range,
                u_wraps,
                v_wraps,
                loops: None,
            }],
        }
    }

    /// The shared construction behind cylinder, cone and frustum.
    ///
    /// `radius_top == 0` collapses the top rim to the apex and drops that cap,
    /// which is what makes a cone a cone rather than a frustum with a degenerate
    /// face.
    fn revolved(base: V3, axis: V3, radius_top: f64, radius_bottom: f64, height: f64) -> Body {
        use std::f64::consts::{PI, TAU};
        const NOMINAL: usize = 16;

        let axis = crate::nurbs::v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
        let side = if (radius_top - radius_bottom).abs() < 1e-12 {
            Surface::cylinder(base, axis, radius_bottom)
        } else {
            let y_apex = height * radius_bottom / (radius_bottom - radius_top);
            let apex = v3::add(base, v3::scale(axis, y_apex));
            let (dir, far_r, far_d) = if radius_bottom > radius_top {
                (v3::scale(axis, -1.0), radius_bottom, y_apex)
            } else {
                (axis, radius_top, height - y_apex)
            };
            match Surface::cone_from_rim(apex, dir, far_r, far_d.abs()) {
                Some(c) => c,
                None => Surface::cylinder(base, axis, radius_bottom.max(radius_top)),
            }
        };
        let top_centre = v3::add(base, v3::scale(axis, height));

        let mut surfaces = vec![side];
        let mut vertices: Vec<V3> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();

        // Bottom rim, always present.
        let bottom_plane = surfaces.len();
        surfaces.push(Surface::plane(base, v3::scale(axis, -1.0)));
        // Cloned rather than borrowed: the closure outlives further pushes to
        // `surfaces`, and a rim only ever needs the side surface.
        let side_surface = surfaces[0].clone();
        let rim = |centre: V3, r: f64, vertices: &mut Vec<V3>| -> Vec<usize> {
            let s = &side_surface;
            let mut ids = Vec::with_capacity(NOMINAL + 1);
            for i in 0..NOMINAL {
                let u = -PI + TAU * i as f64 / NOMINAL as f64;
                // Take the point from the *side* surface, so the rim lies on it
                // exactly rather than merely near it.
                let p = match s.invert(v3::add(
                    centre,
                    v3::scale(crate::nurbs::construct::perpendicular(axis), r),
                )) {
                    Some((_, v)) => s.point(u, v),
                    None => centre,
                };
                vertices.push(p);
                ids.push(vertices.len() - 1);
            }
            ids.push(ids[0]);
            ids
        };

        let bottom = rim(base, radius_bottom, &mut vertices);
        edges.push(Edge {
            surfaces: (0, bottom_plane),
            vertices: bottom,
            closed: true,
        });

        let mut top_plane = None;
        if radius_top > 1e-12 {
            let plane = surfaces.len();
            surfaces.push(Surface::plane(top_centre, axis));
            let top = rim(top_centre, radius_top, &mut vertices);
            edges.push(Edge {
                surfaces: (0, plane),
                vertices: top,
                closed: true,
            });
            top_plane = Some(plane);
        }

        let (v_lo, v_hi) = {
            let s = &side_surface;
            let a = s.invert(vertices[0]).map(|(_, v)| v).unwrap_or(0.0);
            let b = s
                .invert(*vertices.last().unwrap())
                .map(|(_, v)| v)
                .unwrap_or(height);
            // A cone's apex end is not a rim, so extend the footprint to it.
            let apex_v = if radius_top <= 1e-12 { Some(0.0) } else { None };
            let lo = a.min(b).min(apex_v.unwrap_or(f64::INFINITY));
            let hi = a.max(b).max(apex_v.unwrap_or(f64::NEG_INFINITY));
            (lo, hi)
        };

        let mut faces = vec![Face {
            surface: 0,
            edges: (0..edges.len()).collect(),
            flipped: false,
            u_range: (-PI, PI),
            v_range: (v_lo, v_hi),
            u_wraps: true,
            v_wraps: false,
            loops: None,
        }];
        for (plane, edge) in [Some(bottom_plane), top_plane]
            .into_iter()
            .flatten()
            .zip(0..)
        {
            faces.push(Face {
                surface: plane,
                edges: vec![edge],
                flipped: false,
                u_range: (
                    -radius_bottom.max(radius_top),
                    radius_bottom.max(radius_top),
                ),
                v_range: (
                    -radius_bottom.max(radius_top),
                    radius_bottom.max(radius_top),
                ),
                u_wraps: false,
                v_wraps: false,
                loops: None,
            });
        }

        Body {
            surfaces,
            vertices,
            edges,
            faces,
        }
    }
}

/// A closed boundary of a face, in that face's parameter space.
///
/// The structure a B-rep face is actually made of: a face is a surface *plus a
/// trim*, and without loops it can only ever be the whole surface or a
/// rectangle of it. A plate with a hole is the smallest thing that needs them —
/// its top is one plane bounded by an outer rectangle and an inner circle, and
/// no parameter box describes that.
#[derive(Debug, Clone, PartialEq)]
pub struct TrimLoop {
    /// Vertex indices into [`Body::vertices`], in order, not repeating the first.
    pub vertices: Vec<usize>,
    /// Parameters of those vertices on the face's surface.
    pub uv: Vec<[f64; 2]>,
    /// Signed area in parameter space. Positive is the outer loop; a hole runs
    /// the other way.
    pub area: f64,
}

/// What bounds a face in its surface's parameters.
///
/// A `Face` carries both a parameter range and, sometimes, trim loops, and the
/// two are answers to the same question. Which one is authoritative was decided
/// separately at every site that asked, and they disagreed: the tessellator took
/// loops if there were any, `point_on_face` took the first ring that was not a
/// hole, `faces_overlap` did the same, and a whole cylinder wall — bounded by
/// two rims, which are lines with no area — was read three different ways.
///
/// The rule is here now, once. A face is bounded by rings when it has rings that
/// enclose something, and by its parameter rectangle otherwise; a rim is an
/// *edge*, not a footprint.
pub enum Footprint<'a> {
    Rectangle { u: (f64, f64), v: (f64, f64) },
    Rings(&'a [TrimLoop]),
}

impl<'a> Footprint<'a> {
    /// The rule itself: rings when they enclose something, the rectangle when
    /// they do not. Takes the rings rather than the face, so a caller working
    /// from *derived* loops asks the same question as one working from stored
    /// ones — which is the whole point of having it in one place.
    pub fn of(loops: &'a [TrimLoop], u: (f64, f64), v: (f64, f64)) -> Footprint<'a> {
        if loops.iter().any(|l| l.is_outer()) {
            Footprint::Rings(loops)
        } else {
            Footprint::Rectangle { u, v }
        }
    }
}

impl Face {
    /// What this face is bounded by. See [`Footprint`].
    pub fn footprint(&self) -> Footprint<'_> {
        match self.loops.as_deref() {
            Some(ls) => Footprint::of(ls, self.u_range, self.v_range),
            None => Footprint::Rectangle {
                u: self.u_range,
                v: self.v_range,
            },
        }
    }
}

impl TrimLoop {
    /// A ring that bounds no region: a rim, a run along a seam, a sliver.
    ///
    /// The sign of `area` says which way a ring turns and nothing about whether
    /// it encloses anything, and *zero* is the case that matters. A whole
    /// cylinder wall is bounded by its two rims, and a rim is a line at constant
    /// `v` — it has no area at all, so it is neither outer nor a hole, and code
    /// that asks `!is_hole()` for "the face's boundary" gets a line.
    ///
    /// That mistake was made three times independently, in `point_on_face`,
    /// `faces_overlap` and the trimmed fill, and cost two twelve-per-cent wrong
    /// answers before all three were found. It is a question about the ring, so
    /// the ring answers it.
    ///
    /// Scale-free: the area is judged against the ring's own extent, so this
    /// needs no tolerance and means the same thing on a bearing and a bridge.
    pub fn is_degenerate(&self) -> bool {
        if self.uv.len() < 3 {
            return true;
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for c in &self.uv {
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let extent = (hi[0] - lo[0]) * (hi[1] - lo[1]);
        extent <= 0.0 || self.area.abs() <= extent * 1e-9
    }

    /// The ring a face is bounded *by*. Never a rim.
    pub fn is_outer(&self) -> bool {
        !self.is_degenerate() && self.area > 0.0
    }

    /// A ring the face is bounded by from the inside. Never a rim either.
    pub fn is_hole(&self) -> bool {
        !self.is_degenerate() && self.area < 0.0
    }

    /// Whether the ring goes all the way around the surface, in each parameter.
    ///
    /// A closed curve on a closed surface bounds two regions without saying
    /// which — a bore's rim can be read as a *hole* in the tool's whole wall, or
    /// as a *divider* cutting that wall into bands. Both are closed rings and
    /// `area` cannot tell them apart. Measured on a rod bored off-axis, the two
    /// readings give:
    ///
    /// ```text
    /// divider   v [0.00 4.48]  v [4.00 8.00]  v [7.52 12.00]   three bands
    /// hole      v [0.00 12.00] with two holes, and two slivers left over
    /// ```
    ///
    /// Only the first has a band inside the rod, so the second loses the piece
    /// the difference needed and the result was not watertight, by 77 edges.
    ///
    /// The readings are not equally good, and the ring says why: each of those
    /// "holes" ran `u` from 1.23 to 7.44, a span of `2pi`. A hole cannot go the
    /// whole way around the surface it is a hole in. So sum the steps folded
    /// into the period — a ring that closes in the plane totals zero, and one
    /// that wraps totals a period.
    pub fn wraps(&self, period: (Option<f64>, Option<f64>)) -> [bool; 2] {
        let mut out = [false; 2];
        for (k, span) in [period.0, period.1].into_iter().enumerate() {
            let Some(span) = span else { continue };
            // Negated deliberately: `!(span > 0.0)` rejects NaN, where the
            // `span <= 0.0` clippy asks for would wave it through.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(span > 0.0) || self.uv.len() < 2 {
                continue;
            }
            let mut travel = 0.0;
            for i in 0..self.uv.len() {
                let step = self.uv[(i + 1) % self.uv.len()][k] - self.uv[i][k];
                let folded = step - span * (step / span).round();
                travel += folded;
            }
            out[k] = travel.abs() > span * 0.5;
        }
        out
    }

    /// Whether the ring encloses a parameter, brought to the ring's own branch
    /// first.
    ///
    /// `Surface::invert` answers in the canonical period and a ring need not
    /// live there: a sphere cut by a sphere keeps a cavity wall whose `u` runs
    /// from `pi` to `3pi`, so every point of that face inverted into `(-pi, pi]`
    /// and missed its own ring by a whole turn. Nothing was ever *on* the face,
    /// the wall it shared with the tool was never seen as shared, and a second
    /// cut kept both copies and lost an eighth of the solid.
    ///
    /// Asking the ring means no caller has to remember, which is the only reason
    /// the same mistake was not made in more places than it was.
    pub fn contains(&self, p: [f64; 2], period: (Option<f64>, Option<f64>)) -> bool {
        let n = self.uv.len();
        if n < 3 {
            return false;
        }
        let mid = |k: usize| {
            let lo = self.uv.iter().map(|c| c[k]).fold(f64::MAX, f64::min);
            let hi = self.uv.iter().map(|c| c[k]).fold(f64::MIN, f64::max);
            (lo + hi) * 0.5
        };
        let fold = |x: f64, k: usize, period: Option<f64>| match period {
            Some(t) => x - t * ((x - mid(k)) / t).round(),
            None => x,
        };
        let q = [fold(p[0], 0, period.0), fold(p[1], 1, period.1)];
        // Crossing parity, the same test `point_in_ring` runs.
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (self.uv[i], self.uv[j]);
            if (a[1] > q[1]) != (b[1] > q[1])
                && q[0] < (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

impl Body {
    /// A face's trim loops, derived from the edges that bound it.
    ///
    /// The loops are *derived* rather than stored because the edges already
    /// determine them, and two representations of one fact drift apart. The
    /// outer loop is the one with the largest positive area; everything else is
    /// a hole, and its winding is normalized so the fill can rely on it.
    /// A stored ring may name the same vertex more than once, and that is not a
    /// defect. A hole is bridged into its outer ring by running out to it and
    /// back, so the bridge's vertices are traversed twice by construction —
    /// `ball - cross` has a 128-point ring with 63 repeated vertices, at gaps of
    /// 126, 124, 122, … which is that there-and-back read off directly. Two
    /// rims joined into one ring look the same from here.
    ///
    /// So a ring cannot be split where it revisits a vertex, however much it
    /// looks like two rings pinched together. Tried: every one of five
    /// tolerances went from four-of-five resolving to none, 379 to 959 open
    /// edges. Whatever separates a bridge from a genuine pinch, it is not this.
    ///
    /// What a spur must do is leave and return at the *same* vertex, and one
    /// that does not is a real fault — in whatever built the loop, not here.
    /// `ball - bore - cross` at 2e-4 stores this on its sphere:
    ///
    /// ```text
    /// f0  … 528, 370, 688, 689 …  … 690, 689, 688, 371, 372 …
    ///              ^ out at 370             ^ back at 371
    /// ```
    ///
    /// 370 and 371 are neighbours on the bore's rim, so the two legs bound a
    /// thin wedge that no triangle covers, and those legs are exactly two of the
    /// four edges the fill leaves open. `bridge_holes` cannot be the cause: its
    /// splice is `ring[..=b] + hole + ring[b..]`, which repeats *both* ends and
    /// so is a slit of no width at all.
    pub fn face_loops(&self, face_index: usize) -> Vec<TrimLoop> {
        let face = &self.faces[face_index];
        if let Some(given) = &face.loops {
            return given.clone();
        }
        let surface = &self.surfaces[face.surface];

        // Every edge of this face, as segments of vertex indices.
        let mut segments: Vec<(usize, usize)> = Vec::new();
        for &e in &face.edges {
            let v = &self.edges[e].vertices;
            for w in v.windows(2) {
                if w[0] != w[1] {
                    segments.push((w[0], w[1]));
                }
            }
        }
        if segments.is_empty() {
            return Vec::new();
        }

        // Chain them into closed rings.
        let mut used = vec![false; segments.len()];
        let mut rings: Vec<Vec<usize>> = Vec::new();
        for seed in 0..segments.len() {
            if used[seed] {
                continue;
            }
            used[seed] = true;
            let mut ring = vec![segments[seed].0, segments[seed].1];
            loop {
                let tail = *ring.last().unwrap();
                let Some(i) = (0..segments.len())
                    .find(|&i| !used[i] && (segments[i].0 == tail || segments[i].1 == tail))
                else {
                    break;
                };
                used[i] = true;
                let (p, q) = segments[i];
                ring.push(if p == tail { q } else { p });
            }
            if ring.len() > 2 && ring[0] == *ring.last().unwrap() {
                ring.pop();
                rings.push(ring);
            }
        }

        let mut loops: Vec<TrimLoop> = Vec::new();
        for ring in rings {
            let mut uv: Vec<[f64; 2]> = Vec::with_capacity(ring.len());
            let mut ok = true;
            let (periodic_u, periodic_v) = surface.periodic();
            let mut anchor: Option<[f64; 2]> = None;
            for &vi in &ring {
                match surface.invert(self.vertices[vi]) {
                    Some((mut u, mut v)) => {
                        // Unwrap along the ring so a loop crossing the branch cut
                        // stays contiguous rather than folding back on itself.
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
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || uv.len() < 3 {
                continue;
            }
            let area = signed_area(&uv);
            loops.push(TrimLoop {
                vertices: ring,
                uv,
                area,
            });
        }

        // Largest area is the outer loop; make it positive and the rest negative,
        // so `is_hole` means what it says whichever way the input wound.
        if let Some(outer) = loops
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.area.abs().partial_cmp(&b.1.area.abs()).unwrap())
            .map(|(i, _)| i)
        {
            let flip = loops[outer].area < 0.0;
            for (i, l) in loops.iter_mut().enumerate() {
                if flip {
                    l.area = -l.area;
                    l.vertices.reverse();
                    l.uv.reverse();
                }
                if i != outer && l.area > 0.0 {
                    l.area = -l.area;
                    l.vertices.reverse();
                    l.uv.reverse();
                }
            }
        }
        loops
    }
}

/// The value of `x` nearest `anchor`, a whole number of turns away.
fn near(x: f64, anchor: f64) -> f64 {
    use std::f64::consts::TAU;
    x - TAU * ((x - anchor) / TAU).round()
}

fn signed_area(uv: &[[f64; 2]]) -> f64 {
    let n = uv.len();
    let mut a = 0.0;
    for i in 0..n {
        let (p, q) = (uv[i], uv[(i + 1) % n]);
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

fn unwrap_near(x: f64, anchor: f64) -> f64 {
    use std::f64::consts::TAU;
    x - TAU * ((x - anchor) / TAU).round()
}

/// Splice holes into an outer ring so one simple polygon describes the face.
///
/// The repo's `earcut` ignores its `hole_indices` argument — its own comment
/// says bridging "adds ~80 LOC" — so a face with a hole would otherwise come
/// back filled solid, hole and all.
///
/// The classic construction: take each hole's rightmost vertex, cast a ray in
/// `+u`, find the outer edge it first crosses, and pick the bridge endpoint that
/// keeps the splice non-self-intersecting. Holes are processed rightmost-first so
/// a bridge never has to cross one that has not been merged yet.
fn bridge_holes(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
) -> Option<(Vec<[f64; 2]>, Vec<usize>)> {
    // `map` records, for each merged point, which ring it came from and where —
    // the fill needs that to get back to vertex indices.
    let mut ring: Vec<[f64; 2]> = outer.to_vec();
    let mut map: Vec<usize> = (0..outer.len()).collect();
    let mut offsets: Vec<usize> = vec![0];
    for h in holes {
        offsets.push(offsets.last().unwrap() + h.len());
    }

    let mut order: Vec<usize> = (0..holes.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = holes[a].iter().cloned().fold(f64::MIN, |m, p| m.max(p[0]));
        let rb = holes[b].iter().cloned().fold(f64::MIN, |m, p| m.max(p[0]));
        rb.partial_cmp(&ra).unwrap()
    });

    let mut merged: Vec<usize> = Vec::with_capacity(holes.len());
    for hi in order {
        let hole = &holes[hi];
        // Rightmost vertex of the hole.
        let mv = (0..hole.len()).max_by(|&a, &b| hole[a][0].partial_cmp(&hole[b][0]).unwrap())?;
        let m = mv;
        let p = hole[m];

        // The nearest vertex the bridge can actually *reach*.
        //
        // Nearest-to-the-right alone is not enough once there is more than one
        // hole: the ring already contains the holes merged before this one, and
        // the closest vertex may sit on the far side of one, so the bridge runs
        // straight through it. The ring is then self-intersecting and the ear
        // clip fills part of the face and stops — a plate's underside came back
        // covering 336 of the 921 units of area it should have, while its top,
        // whose rings happened to be ordered the other way, was perfect.
        //
        // So test the segment against every edge it must not cross: the ring's
        // own, and those of the holes not yet merged.
        let blocked = |q: [f64; 2], qi: usize| -> bool {
            let n = ring.len();
            for i in 0..n {
                // Edges meeting at `q` share an endpoint and cannot properly
                // cross the segment that ends there.
                if i == qi || (i + 1) % n == qi {
                    continue;
                }
                if crosses(p, q, ring[i], ring[(i + 1) % n]) {
                    return true;
                }
            }
            for (j, other) in holes.iter().enumerate() {
                if merged.contains(&j) {
                    continue; // already part of `ring`, tested above
                }
                let m = other.len();
                for i in 0..m {
                    if j == hi && (i == mv || (i + 1) % m == mv) {
                        continue;
                    }
                    if crosses(p, q, other[i], other[(i + 1) % m]) {
                        return true;
                    }
                }
            }
            false
        };

        // A bridge that arrives *tangentially* is as bad as one that crosses.
        //
        // The rightmost point of a circle has a vertical tangent, so a bridge
        // running vertically into it is collinear with the ring there and the
        // ear at that vertex is exactly flat. The clip then drops it, and with
        // it a rim vertex the bore wall still has — three collinear holes left
        // one face short by a single zero-area triangle, and the seam open
        // along its three edges.
        let flat = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            let scale = ((b[0] - a[0]).hypot(b[1] - a[1])).max(1e-12);
            (cross / scale).abs() < 1e-9
        };
        let tangent = |q: [f64; 2], qi: usize| -> bool {
            let n = ring.len();
            flat(p, ring[(qi + n - 1) % n], q) || flat(p, ring[(qi + 1) % n], q)
        };

        // Candidates: nearest first, but *outer* vertices before hole ones.
        //
        // Bridging one hole to another chains them, and a chain of bridges is
        // what makes the ring only weakly simple — the channels meet at shared
        // vertices, and an ear clip has no guarantee on such a polygon. Left to
        // pick the nearest vertex it will happily chain three holes together and
        // end up clipping a zero-area sliver whose edges pair with nothing.
        //
        // Reaching the outer boundary independently keeps each channel its own.
        let mut candidates: Vec<(bool, f64, usize)> = ring
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let outer_vertex = map[i] < outer.len();
                (
                    !outer_vertex,
                    (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2),
                    i,
                )
            })
            .collect();
        candidates.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        // Rightward first, since that is where a bridge out of a hole normally
        // goes; then anywhere, rather than failing outright.
        let ok = |i: usize| !tangent(ring[i], i) && !blocked(ring[i], i);
        let bridge = candidates
            .iter()
            .find(|&&(_, _, i)| ring[i][0] >= p[0] && ok(i))
            .or_else(|| candidates.iter().find(|&&(_, _, i)| ok(i)))
            // Tangency only costs a triangle; crossing costs the face. If
            // nothing avoids both, take clear-but-tangent over neither.
            .or_else(|| candidates.iter().find(|&&(_, _, i)| !blocked(ring[i], i)))
            .map(|&(_, _, i)| i)?;

        // Splice: outer[..=bridge] + hole rotated to start at m + outer[bridge..].
        let mut spliced: Vec<[f64; 2]> = ring[..=bridge].to_vec();
        let mut spliced_map: Vec<usize> = map[..=bridge].to_vec();
        for k in 0..=hole.len() {
            let idx = (m + k) % hole.len();
            spliced.push(hole[idx]);
            spliced_map.push(outer.len() + offsets[hi] + idx);
        }
        spliced.extend_from_slice(&ring[bridge..]);
        spliced_map.extend_from_slice(&map[bridge..]);
        ring = spliced;
        map = spliced_map;
        merged.push(hi);
    }
    Some((ring, map))
}

/// Whether two segments cross *properly* — sharing an endpoint or merely
/// touching does not count, which is what a bridge needs: it is allowed to land
/// on a vertex, and forbidden to pass through an edge.
fn crosses(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let side = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let (d1, d2) = (side(c, d, a), side(c, d, b));
    let (d3, d4) = (side(a, b, c), side(a, b, d));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

impl Body {
    /// A rectangular plate with a cylindrical hole through it.
    ///
    /// The smallest solid that *needs* trim loops: its top and bottom are each
    /// one plane bounded by an outer rectangle **and** an inner circle, and no
    /// parameter rectangle describes that. Built directly, so it exercises the
    /// loops without going through a boolean.
    ///
    /// The hole runs along `+Z`. Returns `None` if it would not fit inside the
    /// plate.
    pub fn plate_with_hole(size: V3, radius: f64) -> Option<Body> {
        Body::plate_with_holes(size, &[([0.0, 0.0], radius)])
    }

    /// A rectangular plate with several cylindrical holes through it.
    ///
    /// The two-hole case is the one worth having: a face with a *single* hole
    /// needs one bridge to triangulate, and a bridge to the outer boundary is
    /// always available. With two, the second bridge may have to run to the
    /// first hole rather than to the boundary, and whether that segment is
    /// clear is a question the single-hole case never asks.
    ///
    /// Holes run along `+Z`, positioned by their centre in the XY plane.
    /// `None` if any of them would not fit inside the plate or would overlap
    /// another.
    pub fn plate_with_holes(size: V3, holes: &[([f64; 2], f64)]) -> Option<Body> {
        use std::f64::consts::{PI, TAU};
        const NOMINAL: usize = 16;

        let (hx, hy, hz) = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5);
        if holes.is_empty() {
            return None;
        }
        for (i, &(c, r)) in holes.iter().enumerate() {
            // `!(r > 0.0)` rather than `r <= 0.0`: the negated form is true for
            // NaN, which the positive one lets through.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(r > 0.0) || c[0].abs() + r >= hx || c[1].abs() + r >= hy {
                return None;
            }
            for &(d, s) in &holes[i + 1..] {
                if ((c[0] - d[0]).powi(2) + (c[1] - d[1]).powi(2)).sqrt() <= r + s {
                    return None;
                }
            }
        }

        // Six plates, then the bore.
        let mut surfaces = vec![
            Surface::plane([hx, 0.0, 0.0], [1.0, 0.0, 0.0]),
            Surface::plane([-hx, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            Surface::plane([0.0, hy, 0.0], [0.0, 1.0, 0.0]),
            Surface::plane([0.0, -hy, 0.0], [0.0, -1.0, 0.0]),
            Surface::plane([0.0, 0.0, hz], [0.0, 0.0, 1.0]),
            Surface::plane([0.0, 0.0, -hz], [0.0, 0.0, -1.0]),
        ];
        // A bore's outward normal points *into* the material, so the face that
        // uses it is flipped — a hole is a surface seen from the other side.
        let first_bore = surfaces.len();
        for &(c, r) in holes {
            surfaces.push(Surface::cylinder([c[0], c[1], 0.0], [0.0, 0.0, 1.0], r));
        }

        let corner = |i: usize| -> V3 {
            [
                if i & 1 == 0 { -hx } else { hx },
                if i & 2 == 0 { -hy } else { hy },
                if i & 4 == 0 { -hz } else { hz },
            ]
        };
        let mut vertices: Vec<V3> = (0..8).map(corner).collect();

        let box_spec: [(usize, usize, usize, usize); 12] = [
            (0, 2, 3, 7),
            (0, 3, 1, 5),
            (0, 4, 5, 7),
            (0, 5, 1, 3),
            (1, 2, 2, 6),
            (1, 3, 0, 4),
            (1, 4, 4, 6),
            (1, 5, 0, 2),
            (2, 4, 6, 7),
            (2, 5, 2, 3),
            (3, 4, 4, 5),
            (3, 5, 0, 1),
        ];
        let mut edges: Vec<Edge> = box_spec
            .iter()
            .map(|&(a, b, v0, v1)| Edge {
                surfaces: (a.min(b), a.max(b)),
                vertices: vec![v0, v1],
                closed: false,
            })
            .collect();

        // Two rims per bore.
        for b in 0..holes.len() {
            let bore = first_bore + b;
            for (plane, z) in [(4usize, hz), (5usize, -hz)] {
                let mut ids = Vec::with_capacity(NOMINAL + 1);
                for i in 0..NOMINAL {
                    let u = -PI + TAU * i as f64 / NOMINAL as f64;
                    vertices.push(surfaces[bore].point(u, z));
                    ids.push(vertices.len() - 1);
                }
                ids.push(ids[0]);
                edges.push(Edge {
                    surfaces: (plane.min(bore), plane.max(bore)),
                    vertices: ids,
                    closed: true,
                });
            }
        }

        let big = hx.max(hy).max(hz) * 2.0;
        let mut faces: Vec<Face> = (0..6)
            .map(|si| Face {
                surface: si,
                edges: (0..edges.len())
                    .filter(|&e| edges[e].surfaces.0 == si || edges[e].surfaces.1 == si)
                    .collect(),
                flipped: false,
                u_range: (-big, big),
                v_range: (-big, big),
                u_wraps: false,
                v_wraps: false,
                loops: None,
            })
            .collect();
        for b in 0..holes.len() {
            let bore = first_bore + b;
            faces.push(Face {
                surface: bore,
                edges: (0..edges.len())
                    .filter(|&e| edges[e].surfaces.0 == bore || edges[e].surfaces.1 == bore)
                    .collect(),
                // Seen from inside the material.
                flipped: true,
                u_range: (-PI, PI),
                v_range: (-hz, hz),
                u_wraps: true,
                v_wraps: false,
                loops: None,
            });
        }

        Some(Body::from_parts(surfaces, vertices, edges, faces))
    }
}

/// Ear-clip a simple polygon that may contain bridge channels.
///
/// The repo's `curves::earcut` is a plain ear clip and gives up on the polygon
/// hole bridging produces: a bridge is a zero-width channel, so the ring has two
/// coincident edges and several exactly-degenerate ears. That is not a defect in
/// it — nothing else feeds it such a polygon — but it is why a face with a hole
/// came back unfilled.
///
/// This one differs in three ways, each needed for that input:
///
/// * **Zero-area ears are clipped, not rejected.** The bridge produces them by
///   construction, and refusing them stalls the clip with vertices left over.
/// * **Containment excludes the ear's own vertices**, so a coincident bridge
///   vertex does not veto every ear that touches it.
/// * **It cannot loop for ever.** If a full pass clips nothing, the remaining
///   fan is emitted rather than spinning — a wrong triangle is recoverable, a
///   hang is not.
fn earclip(points: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }
    let area2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    };
    // Work counter-clockwise so an ear is a positive turn.
    let mut idx: Vec<usize> = (0..n).collect();
    if signed_area(points) < 0.0 {
        idx.reverse();
    }

    // Strictly inside, and not a copy of one of the ear's own corners.
    //
    // Inclusive containment blocks every ear here: a bridge duplicates a
    // position, so points sit exactly on ear edges by construction and an
    // `>= 0` test vetoes all of them — the clip then makes no progress and
    // returns nothing, which is precisely how a holed face came back unfilled.
    let same =
        |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() < 1e-12 && (p[1] - q[1]).abs() < 1e-12;
    let inside = |a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]| {
        if same(p, a) || same(p, b) || same(p, c) {
            return false;
        }
        area2(a, b, p) > 0.0 && area2(b, c, p) > 0.0 && area2(c, a, p) > 0.0
    };

    // An ear whose base runs *along* the boundary is not an ear.
    //
    // Clipping (a, b, c) draws the edge a-c, and the fan a clip makes can put
    // that edge between two ring vertices far apart along the same ring. Where
    // the ring is a rim — every point of it at one `v` on a cylinder, one
    // latitude on a sphere — that chord lies on the boundary in parameter space
    // however long it is. The face beyond the rim triangulates it with the rim's
    // own points and has no such edge, so it is left open. Measured on a rod
    // added to a bored ball: a single edge of length 1.56 between two points
    // both on the rod's wall and both on the sphere, spanning 112 degrees of a
    // circle of radius 0.8.
    //
    // `refine_on_surface` already refuses to *split* these, and says the cure is
    // not to create them. This is that: an ear is refused if its base lies along
    // the ring between two vertices that are not neighbours on it.
    let extent = {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for q in points {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12)
    };
    let near = extent * 1e-9;
    let along_ring = |ia: usize, ic: usize| -> bool {
        if (ia + 1) % n == ic || (ic + 1) % n == ia {
            return false; // neighbours: the base *is* one step of the ring
        }
        let mid = [
            0.5 * (points[ia][0] + points[ic][0]),
            0.5 * (points[ia][1] + points[ic][1]),
        ];
        (0..n).any(|k| {
            let (x, y) = (points[k], points[(k + 1) % n]);
            let (dx, dy) = (y[0] - x[0], y[1] - x[1]);
            let len2 = dx * dx + dy * dy;
            let t = if len2 <= f64::MIN_POSITIVE {
                0.0
            } else {
                (((mid[0] - x[0]) * dx + (mid[1] - x[1]) * dy) / len2).clamp(0.0, 1.0)
            };
            (mid[0] - (x[0] + dx * t)).hypot(mid[1] - (x[1] + dy * t)) <= near
        })
    };

    let mut out = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0usize;
    while idx.len() > 3 {
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (points[ia], points[ib], points[ic]);
            let cross = area2(a, b, c);
            if cross < 0.0 {
                continue; // reflex
            }
            if cross > 0.0 {
                if along_ring(ia, ic) {
                    continue;
                }
                let blocked = idx
                    .iter()
                    .any(|&j| j != ia && j != ib && j != ic && inside(a, b, c, points[j]));
                if blocked {
                    continue;
                }
            }
            // Emitted whether flat or not.
            //
            // A flat ear draws nothing, so clipping one without emitting just
            // deletes its middle vertex — and on a swept face that vertex is
            // usually a *rim* point, since a rim is a straight line in parameter
            // space and every ear along it is exactly flat. Eleven of sixteen
            // rim vertices of a bored cylinder's wall were being deleted this
            // way, and the cap beyond the rim was left holding edges to them.
            //
            // A zero-area triangle draws nothing either, and keeps the vertex
            // attached.
            out.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        guard += 1;
        if !clipped || guard > n * n + 8 {
            break;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}
