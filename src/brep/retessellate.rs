//! Re-mesh a geometry from its provenance at an arbitrary tolerance.
//!
//! The payoff that motivates carrying surfaces at all: `$fn` no longer has to be
//! chosen before the boolean and lived with forever. Boolean coarse, render
//! fine; mesh a planet's shell at screen tolerance and its print at 0.05 mm,
//! from the same solid.
//!
//! # What this does not do
//!
//! Each surface patch is re-meshed **independently**, over the parameter range
//! its own triangles occupy. Adjacent patches therefore need not agree along
//! their shared boundary, and where they do not the result has cracks —
//! a cylinder's side sampled at 40 angular steps against a cap rim sampled at
//! 31 leaves a ring of T-junctions.
//!
//! Making patches share an edge is *trimming*, which needs `Loop`/`Edge`
//! topology and is Stage 3. Until then this reports what it produced rather
//! than implying more: [`RetessellationReport::boundary_edges`] counts edges
//! used by exactly one triangle, so a caller can tell a closed result from a
//! cracked one instead of discovering it in a slicer.
//!
//! Single-surface geometry — spheres, tori, cylinder sides, any one NURBS patch
//! — has no patch boundaries and comes back closed.

use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::nurbs::v3;
use crate::nurbs::V3;

use super::table::{triangle_indices, triangle_normal, triangle_vertices};
use super::{Surface, SurfaceTable};

/// What a re-tessellation actually produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetessellationReport {
    /// Surface patches re-meshed.
    pub patches: usize,
    pub vertices: usize,
    pub triangles: usize,
    /// Edges used by exactly one triangle after welding.
    ///
    /// Zero means closed. Non-zero means patch boundaries did not line up,
    /// which is expected for multi-surface input and is reported rather than
    /// hidden — a cracked mesh looks identical to a closed one until something
    /// downstream needs it to be watertight.
    pub boundary_edges: usize,
    /// Patches whose surface could not be sampled (a degenerate inversion).
    /// Their original triangles are carried through unchanged.
    pub carried_through: usize,
}

impl RetessellationReport {
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0
    }
}

/// Re-mesh `geometry` from its attached provenance at `tolerance` model units.
///
/// Returns `None` when there is no usable provenance — the caller keeps the mesh
/// it already has, which is always correct.
pub fn retessellate(
    geometry: &BufferGeometry,
    tolerance: f64,
) -> Option<(BufferGeometry, RetessellationReport)> {
    let table = geometry.surface_table()?;
    // The shared boundaries, recovered once. Both faces meeting along one look
    // it up, which is what makes them agree about where it is.
    //
    // Left at the input mesh's resolution deliberately. Densifying them to the
    // requested tolerance first *sounds* right — the side wants ~124 samples
    // around a rim the input gives 24 — but measured, it made things worse: the
    // cone went from a closed result to 118 open edges, because the two faces
    // then arrive at nearby-but-not-equal points from different arithmetic and
    // the weld no longer merges them. Closing that properly needs the faces to
    // share *vertices*, not just parameters, which is the topology Stage 3b
    // brings.
    let boundaries = super::stitch::patch_boundaries(geometry, table);

    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut uvs: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut tri_face: Vec<u32> = Vec::new();

    let mut weld: HashMap<[i64; 3], u32> = HashMap::new();
    let mut patches = 0usize;
    let mut carried_through = 0usize;

    // Weld quantum: fine enough not to merge distinct vertices of a mesh at this
    // tolerance, coarse enough to catch the same corner arrived at from two
    // patches after different arithmetic.
    let quantum = (tolerance * 1e-3).max(1e-12);

    for (surface_index, triangles) in table.groups() {
        let surface = &table.surfaces()[surface_index];
        patches += 1;

        // A planar patch is filled against its boundary; a curved one keeps the
        // grid but samples the boundary's parameters. See the module docs.
        if matches!(surface, Surface::Plane { .. })
            && emit_planar_patch(
                surface,
                surface_index,
                &boundaries,
                quantum,
                &mut weld,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                &mut tri_face,
                geometry,
                &triangles,
            )
        {
            continue;
        }

        match patch_grid(
            geometry,
            surface,
            &triangles,
            tolerance,
            &boundaries,
            surface_index,
        ) {
            Some(patch) => emit_patch(
                &patch,
                surface,
                surface_index as u32,
                quantum,
                &mut weld,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
                &mut tri_face,
            ),
            None => {
                carried_through += 1;
                for &tri in &triangles {
                    let (Some(verts), Some(_)) = (
                        triangle_vertices(geometry, tri),
                        triangle_indices(geometry, tri),
                    ) else {
                        continue;
                    };
                    let n = triangle_normal(&verts).unwrap_or([0.0, 0.0, 1.0]);
                    for p in verts {
                        let vi = weld_vertex(
                            p,
                            n,
                            [0.0, 0.0],
                            quantum,
                            &mut weld,
                            &mut positions,
                            &mut normals,
                            &mut uvs,
                        );
                        indices.push(vi);
                    }
                    tri_face.push(surface_index as u32);
                }
            }
        }
    }

    let boundary_edges = count_boundary_edges(&indices);

    let mut out = BufferGeometry::new();
    let vertices = positions.len() / 3;
    out.set_attribute("position", BufferAttribute::new(positions, 3));
    out.set_attribute("normal", BufferAttribute::new(normals, 3));
    out.set_attribute("uv", BufferAttribute::new(uvs, 2));
    out.set_index(indices.clone());
    if let Some(t) = SurfaceTable::new(table.surfaces().to_vec(), tri_face) {
        out.set_surfaces(t);
    }

    let report = RetessellationReport {
        patches,
        vertices,
        triangles: indices.len() / 3,
        boundary_edges,
        carried_through,
    };
    Some((out, report))
}

/// A patch's sample grid in parameter space, plus whether its winding opposes
/// the surface's canonical sense.
struct PatchGrid {
    us: Vec<f64>,
    vs: Vec<f64>,
    flipped: bool,
}

/// Work out the parameter range a patch's triangles occupy and how densely to
/// sample it.
#[allow(clippy::too_many_arguments)]
fn patch_grid(
    geometry: &BufferGeometry,
    surface: &Surface,
    triangles: &[usize],
    tolerance: f64,
    boundaries: &[super::stitch::Boundary],
    surface_index: usize,
) -> Option<PatchGrid> {
    use std::f64::consts::TAU;
    let (periodic_u, periodic_v) = surface.periodic();

    let (mut u0, mut u1) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut flip_votes = 0i32;
    let mut samples = 0usize;

    for &tri in triangles {
        let verts = triangle_vertices(geometry, tri)?;
        if let (Some(face), Some((cu, cv))) =
            (triangle_normal(&verts), surface.invert(centroid(&verts)))
        {
            if let Some(sn) = surface.normal(cu, cv) {
                flip_votes += if v3::dot(face, sn) < 0.0 { -1 } else { 1 };
            }
        }
        for p in verts {
            let Some((u, v)) = surface.invert(p) else {
                continue;
            };
            u0 = u0.min(u);
            u1 = u1.max(u);
            v0 = v0.min(v);
            v1 = v1.max(v);
            samples += 1;
        }
    }
    if samples == 0 || !u0.is_finite() || !v0.is_finite() {
        return None;
    }

    // A periodic direction whose samples span most of a turn is a full wrap: the
    // observed range stops just short of 2π because there is no vertex *at* the
    // seam, and sampling that literal range would leave a wedge missing.
    if periodic_u && u1 - u0 > TAU * 0.75 {
        u0 = -std::f64::consts::PI;
        u1 = std::f64::consts::PI;
    }
    if periodic_v && v1 - v0 > TAU * 0.75 {
        v0 = -std::f64::consts::PI;
        v1 = std::f64::consts::PI;
    }
    if u1 - u0 < 1e-12 || v1 - v0 < 1e-12 {
        return None; // a degenerate patch has no grid
    }

    // Each direction is sampled to *half* the requested tolerance.
    //
    // The caller asked for a mesh within `tolerance`, and a mesh has diagonals:
    // a quad's diagonal spans both parameter directions at once, so its chord
    // deviation is bounded by the sum of the two per-direction deviations, not
    // by either alone. Sampling each to `tolerance` would leave every diagonal
    // at up to `2·tolerance` — measurably outside the number that was asked for,
    // and invisible unless you go looking. Half costs about √2 more samples per
    // direction and makes the guarantee the one the signature implies.
    let half = tolerance * 0.5;
    let mut us = sample_direction(surface, (u0, u1), (v0, v1), half, true);
    let mut vs = sample_direction(surface, (u0, u1), (v0, v1), half, false);

    // Fold in the parameters of every boundary this patch shares. Without this
    // the two faces meeting along a rim sample it at different places and the
    // result is a ring of T-junctions — they agree on *where* the boundary is
    // and disagree on which points along it to use, and a mesh is made of points.
    let (mut bu, mut bv): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for b in boundaries {
        if b.surfaces.0 != surface_index && b.surfaces.1 != surface_index {
            continue;
        }
        for p in &b.points {
            let Some((pu, pv)) = surface.invert(*p) else {
                continue;
            };
            if pu >= u0 - 1e-9 && pu <= u1 + 1e-9 {
                bu.push(pu);
            }
            if pv >= v0 - 1e-9 && pv <= v1 + 1e-9 {
                bv.push(pv);
            }
        }
    }
    let tidy = |v: &mut Vec<f64>, lo: f64, hi: f64| {
        v.retain(|x| x.is_finite());
        v.push(lo);
        v.push(hi);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup_by(|a, b| (*a - *b).abs() <= (hi - lo) * 1e-9);
    };
    us.extend(bu);
    vs.extend(bv);
    tidy(&mut us, u0, u1);
    tidy(&mut vs, v0, v1);

    Some(PatchGrid {
        us,
        vs,
        flipped: flip_votes < 0,
    })
}

/// Sample parameters along one direction so the chord deviation stays within
/// `tolerance`, measured along several iso-lines of the other.
///
/// Bisection rather than a closed-form segment count: it works for every variant
/// including NURBS, where there is no radius to derive a count from.
fn sample_direction(
    surface: &Surface,
    (u0, u1): (f64, f64),
    (v0, v1): (f64, f64),
    tolerance: f64,
    along_u: bool,
) -> Vec<f64> {
    const PROBES: usize = 5;
    const MAX_DEPTH: usize = 12;

    let (a, b) = if along_u { (u0, u1) } else { (v0, v1) };
    let (o0, o1) = if along_u { (v0, v1) } else { (u0, u1) };
    let probes: Vec<f64> = (0..PROBES)
        .map(|i| o0 + (o1 - o0) * i as f64 / (PROBES - 1).max(1) as f64)
        .collect();

    let at = |t: f64| -> Vec<V3> {
        probes
            .iter()
            .map(|&o| {
                if along_u {
                    surface.point(t, o)
                } else {
                    surface.point(o, t)
                }
            })
            .collect()
    };

    let mut out = vec![a];
    split(a, b, 0, MAX_DEPTH, tolerance, &at, &mut out);
    out.push(b);
    out
}

fn split(
    a: f64,
    b: f64,
    depth: usize,
    max_depth: usize,
    tolerance: f64,
    at: &dyn Fn(f64) -> Vec<V3>,
    out: &mut Vec<f64>,
) {
    if depth >= max_depth {
        return;
    }
    let (pa, pb) = (at(a), at(b));
    let mut worst = 0.0f64;
    for k in 1..4 {
        let t = k as f64 / 4.0;
        let pm = at(a + (b - a) * t);
        for ((qa, qb), qm) in pa.iter().zip(pb.iter()).zip(pm.iter()) {
            let chord = v3::add(v3::scale(*qa, 1.0 - t), v3::scale(*qb, t));
            worst = worst.max(v3::dist(*qm, chord));
        }
    }
    if worst <= tolerance {
        return;
    }
    let m = 0.5 * (a + b);
    split(a, m, depth + 1, max_depth, tolerance, at, out);
    out.push(m);
    split(m, b, depth + 1, max_depth, tolerance, at, out);
}

#[allow(clippy::too_many_arguments)]
fn emit_patch(
    patch: &PatchGrid,
    surface: &Surface,
    surface_index: u32,
    quantum: f64,
    weld: &mut HashMap<[i64; 3], u32>,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    tri_face: &mut Vec<u32>,
) {
    let (nu, nv) = (patch.us.len(), patch.vs.len());
    let (uspan, vspan) = (
        (patch.us[nu - 1] - patch.us[0]).max(f64::MIN_POSITIVE),
        (patch.vs[nv - 1] - patch.vs[0]).max(f64::MIN_POSITIVE),
    );

    let mut grid = vec![0u32; nu * nv];
    for (i, &u) in patch.us.iter().enumerate() {
        for (j, &v) in patch.vs.iter().enumerate() {
            let p = surface.point(u, v);
            let mut n = surface.normal(u, v).unwrap_or([0.0, 0.0, 1.0]);
            if patch.flipped {
                n = v3::scale(n, -1.0);
            }
            let uv = [
                ((u - patch.us[0]) / uspan) as f32,
                ((v - patch.vs[0]) / vspan) as f32,
            ];
            grid[i * nv + j] = weld_vertex(p, n, uv, quantum, weld, positions, normals, uvs);
        }
    }

    for i in 0..nu - 1 {
        for j in 0..nv - 1 {
            let (a, b, c, d) = (
                grid[i * nv + j],
                grid[(i + 1) * nv + j],
                grid[(i + 1) * nv + j + 1],
                grid[i * nv + j + 1],
            );
            let pos = |k: u32| {
                let o = k as usize * 3;
                [
                    positions[o] as f64,
                    positions[o + 1] as f64,
                    positions[o + 2] as f64,
                ]
            };
            // The surface's own (u, v) winding is counter-clockwise seen from
            // +normal; `flipped` reverses it to match the input mesh.
            let (t0, t1) = if patch.flipped {
                ([a, c, b], [a, d, c])
            } else {
                ([a, b, c], [a, c, d])
            };
            for t in [t0, t1] {
                if !degenerate(pos(t[0]), pos(t[1]), pos(t[2])) {
                    indices.extend_from_slice(&t);
                    tri_face.push(surface_index);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn weld_vertex(
    p: V3,
    n: V3,
    uv: [f32; 2],
    quantum: f64,
    weld: &mut HashMap<[i64; 3], u32>,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
) -> u32 {
    let key = [
        (p[0] / quantum).round() as i64,
        (p[1] / quantum).round() as i64,
        (p[2] / quantum).round() as i64,
    ];
    if let Some(&existing) = weld.get(&key) {
        return existing;
    }
    let vi = (positions.len() / 3) as u32;
    positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
    normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);
    uvs.extend_from_slice(&uv);
    weld.insert(key, vi);
    vi
}

fn centroid(v: &[V3; 3]) -> V3 {
    [
        (v[0][0] + v[1][0] + v[2][0]) / 3.0,
        (v[0][1] + v[1][1] + v[2][1]) / 3.0,
        (v[0][2] + v[1][2] + v[2][2]) / 3.0,
    ]
}

fn degenerate(a: V3, b: V3, c: V3) -> bool {
    let ab = v3::sub(b, a);
    let ac = v3::sub(c, a);
    let area2 = v3::norm(v3::cross(ab, ac));
    let scale = v3::norm(ab).max(v3::norm(ac)).max(v3::dist(b, c));
    area2 <= 1e-12 * scale * scale
}

/// Edges used by exactly one triangle. Zero means every edge is shared, which
/// for an orientable mesh means closed.
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


/// Fill a planar patch against its actual boundary rather than its parameter
/// bounding box.
///
/// A plane is exact under any triangulation, so filling the boundary polygon
/// directly is not an approximation — it is the whole answer, and it is the only
/// way a non-rectangular face (a cylinder's disk cap) comes back as itself
/// instead of as the square its parameters span.
///
/// Returns `false` if the patch has no usable boundary, so the caller falls back
/// to the grid — which is right for a planar patch that genuinely *is* its own
/// bounding box, and for one whose boundary the mesh does not reveal.
#[allow(clippy::too_many_arguments)]
fn emit_planar_patch(
    surface: &Surface,
    surface_index: usize,
    boundaries: &[super::stitch::Boundary],
    quantum: f64,
    weld: &mut HashMap<[i64; 3], u32>,
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    tri_face: &mut Vec<u32>,
    geometry: &BufferGeometry,
    triangles: &[usize],
) -> bool {
    use crate::math::Vector2;

    // The patch's own boundary loops, in order.
    let loops: Vec<&super::stitch::Boundary> = boundaries
        .iter()
        .filter(|b| (b.surfaces.0 == surface_index || b.surfaces.1 == surface_index) && b.closed)
        .collect();
    // Exactly one closed loop is the case this handles. Several would be a face
    // with holes, which needs a hole-bridging fill; none means the mesh did not
    // reveal a boundary. Both fall back rather than guess.
    if loops.len() != 1 {
        return false;
    }
    let ring = &loops[0].points;
    if ring.len() < 4 {
        return false;
    }

    // Project into the plane. The last point repeats the first on a closed loop.
    let mut uv: Vec<(f64, f64)> = Vec::with_capacity(ring.len() - 1);
    for p in &ring[..ring.len() - 1] {
        let Some(q) = surface.invert(*p) else {
            return false;
        };
        uv.push(q);
    }

    // Which way does the mesh wind this face? The fill has to match, or the
    // patch comes back inside-out.
    let flipped = {
        let mut votes = 0i32;
        for &tri in triangles {
            let Some(v) = triangle_vertices(geometry, tri) else {
                continue;
            };
            if let (Some(face), Some((cu, cv))) =
                (triangle_normal(&v), surface.invert(centroid(&v)))
            {
                if let Some(sn) = surface.normal(cu, cv) {
                    votes += if v3::dot(face, sn) < 0.0 { -1 } else { 1 };
                }
            }
        }
        votes < 0
    };

    let pts2: Vec<Vector2> = uv
        .iter()
        .map(|&(u, v)| Vector2::new(u as f32, v as f32))
        .collect();
    let tris = crate::curves::earcut::earcut(&pts2, &[]);
    if tris.is_empty() {
        return false;
    }

    // Emit against the *3D* ring points, not re-evaluated ones: these are the
    // vertices the neighbouring patch also uses, and welding them by position is
    // what closes the seam.
    let ids: Vec<u32> = ring[..ring.len() - 1]
        .iter()
        .zip(&uv)
        .map(|(p, &(u, v))| {
            let mut n = surface.normal(u, v).unwrap_or([0.0, 0.0, 1.0]);
            if flipped {
                n = v3::scale(n, -1.0);
            }
            weld_vertex(*p, n, [0.0, 0.0], quantum, weld, positions, normals, uvs)
        })
        .collect();

    for t in tris.chunks_exact(3) {
        let (a, b, c) = (ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]);
        let pos = |k: u32| {
            let o = k as usize * 3;
            [
                positions[o] as f64,
                positions[o + 1] as f64,
                positions[o + 2] as f64,
            ]
        };
        if degenerate(pos(a), pos(b), pos(c)) {
            continue;
        }
        // `earcut` winds counter-clockwise in the plane's own coordinates, which
        // is `+normal`; reverse it where the mesh disagrees.
        if flipped {
            indices.extend_from_slice(&[a, c, b]);
        } else {
            indices.extend_from_slice(&[a, b, c]);
        }
        tri_face.push(surface_index as u32);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::{BoxGeometry, CylinderGeometry, SphereGeometry};

    const TAU: f32 = std::f32::consts::PI * 2.0;

    fn tagged_sphere(radius: f64, w: usize, h: usize) -> BufferGeometry {
        let mut g = SphereGeometry::new(radius as f32, w, h);
        let tris = crate::brep::triangle_count(&g);
        g.set_surfaces(SurfaceTable::uniform(
            Surface::sphere([0.0; 3], radius),
            tris,
        ));
        g
    }

    fn max_radial_error(g: &BufferGeometry, radius: f64) -> f64 {
        g.get_attribute("position")
            .unwrap()
            .array
            .chunks_exact(3)
            .map(|c| {
                let p = [c[0] as f64, c[1] as f64, c[2] as f64];
                (v3::norm(p) - radius).abs()
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn a_coarse_sphere_retessellates_to_tolerance() {
        // The plan's acceptance criterion: a $fn=16-equivalent sphere re-meshed
        // at 1e-4 is within 1e-4 of the analytic sphere.
        let coarse = tagged_sphere(1.0, 16, 8);
        let before = chord_error(&coarse, 1.0);
        assert!(
            before > 1e-2,
            "coarse mesh should be visibly coarse: {before}"
        );

        let (fine, report) = retessellate(&coarse, 1e-4).expect("provenance is attached");
        assert_eq!(report.patches, 1);
        // Positions are f32 in the buffer, so the floor is f32 epsilon times the
        // radius (~1e-7), not f64 exactness.
        assert!(
            max_radial_error(&fine, 1.0) < 1e-6,
            "vertices left the sphere"
        );

        let after = chord_error(&fine, 1.0);
        assert!(after <= 1e-4, "chord error {after} exceeds the tolerance");
        assert!(after < before / 100.0, "{after} vs {before}");
    }

    /// Worst deviation of an edge midpoint from the sphere — the mesh's real
    /// error, which its vertices (all exactly on the sphere) do not show.
    fn chord_error(g: &BufferGeometry, radius: f64) -> f64 {
        let pos = &g.get_attribute("position").unwrap().array;
        let idx = g.index.as_ref().unwrap();
        let mut worst = 0.0f64;
        for t in idx.chunks_exact(3) {
            let p: Vec<V3> = t
                .iter()
                .map(|&i| {
                    let o = i as usize * 3;
                    [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
                })
                .collect();
            for k in 0..3 {
                let mid = v3::scale(v3::add(p[k], p[(k + 1) % 3]), 0.5);
                worst = worst.max((radius - v3::norm(mid)).abs());
            }
        }
        worst
    }

    #[test]
    fn a_single_surface_patch_comes_back_closed() {
        let (_out, report) = retessellate(&tagged_sphere(2.0, 24, 12), 1e-3).unwrap();
        assert_eq!(report.carried_through, 0);
        assert!(
            report.is_closed(),
            "a sphere has no patch boundaries but reported {} open edges",
            report.boundary_edges
        );
    }

    #[test]
    fn retessellation_carries_provenance_forward() {
        let (out, _) = retessellate(&tagged_sphere(1.0, 12, 8), 1e-3).unwrap();
        let t = out.surface_table().expect("the result is still tagged");
        assert_eq!(t.surfaces().len(), 1);
        // f32 storage, again — 1e-9 would be asking the buffer for precision it
        // does not have.
        assert!(
            t.max_deviation(&out) < 1e-6,
            "re-meshed vertices are off-surface"
        );

        // And it can be done again, from the result.
        let (again, _) = retessellate(&out, 1e-2).unwrap();
        assert!(again.surface_table().is_some());
    }

    #[test]
    fn coarser_tolerance_produces_fewer_triangles() {
        let g = tagged_sphere(1.0, 16, 8);
        let (fine, fr) = retessellate(&g, 1e-4).unwrap();
        let (coarse, cr) = retessellate(&g, 1e-2).unwrap();
        assert!(
            cr.triangles < fr.triangles,
            "coarse {} vs fine {}",
            cr.triangles,
            fr.triangles
        );
        assert!(chord_error(&coarse, 1.0) <= 1e-2);
        assert!(chord_error(&fine, 1.0) <= 1e-4);
    }

    #[test]
    fn a_disk_cap_comes_back_as_a_disk_not_as_its_bounding_square() {
        // A patch's parameter *footprint* is not in general a rectangle, and
        // meshing its bounding box instead is a wrong answer rather than a
        // coarse one. A radius-2 cap used to come back reaching 2.83 — the
        // corner of the [-2, 2]² box its parameters span.
        let g = CylinderGeometry::new(2.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let (out, report) = retessellate(&g, 1e-3).unwrap();
        assert_eq!(report.patches, 3);

        let mut worst = 0.0f64;
        for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
            worst = worst.max(((c[0] as f64).powi(2) + (c[2] as f64).powi(2)).sqrt());
        }
        assert!(
            worst <= 2.0 + 1e-4,
            "a vertex reached radius {worst}; the solid's is 2.0"
        );

        // And every vertex is still on a surface the table claims.
        assert!(out.surface_table().unwrap().max_deviation(&out) < 1e-4);
    }

    #[test]
    fn a_cone_retessellates_to_the_right_shape_and_reports_its_seams() {
        // Geometry: correct. Closure: not guaranteed here, and not claimed.
        //
        // This path re-meshes each patch from its parameter footprint and welds
        // by position, which cannot make two faces agree on *which* points to
        // put along a shared rim. `brep::Body` can, because its faces share
        // vertex indices rather than positions — and it closes this same cone.
        let g = CylinderGeometry::new(0.0, 2.0, 4.0, 24, 1, false, 0.0, TAU);
        let (out, report) = retessellate(&g, 1e-3).unwrap();
        assert_eq!(report.patches, 2);

        for c in out.get_attribute("position").unwrap().array.chunks_exact(3) {
            let r = ((c[0] as f64).powi(2) + (c[2] as f64).powi(2)).sqrt();
            assert!(r <= 2.0 + 1e-4, "radius {r}");
        }
        assert!(out.surface_table().unwrap().max_deviation(&out) < 1e-4);
        // Whatever it produced, the report agrees with itself.
        assert_eq!(report.is_closed(), report.boundary_edges == 0);
    }

    #[test]
    fn a_box_of_planes_round_trips_and_stays_closed() {
        // Six planar patches. Planes are exact at any sample count, and their
        // shared edges are exact corners, so the weld closes them.
        let mut g = BoxGeometry::new(2.0, 3.0, 4.0);
        let tris = crate::brep::triangle_count(&g);
        let planes: Vec<Surface> = [
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            ([0.0, 1.5, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, -1.5, 0.0], [0.0, -1.0, 0.0]),
            ([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, -2.0], [0.0, 0.0, -1.0]),
        ]
        .iter()
        .map(|&(o, n)| Surface::plane(o, n))
        .collect();
        let tri_face: Vec<u32> = (0..tris as u32).map(|i| i / 2).collect();
        g.set_surfaces(SurfaceTable::new(planes, tri_face).unwrap());

        let (out, report) = retessellate(&g, 1e-3).unwrap();
        assert_eq!(report.patches, 6);
        assert_eq!(report.carried_through, 0);
        assert!(
            report.is_closed(),
            "planar patches meeting at exact corners should weld closed, got {} open edges",
            report.boundary_edges
        );

        // The result is the same box.
        let pos = &out.get_attribute("position").unwrap().array;
        for c in pos.chunks_exact(3) {
            assert!((c[0].abs() - 1.0).abs() < 1e-5 || c[0].abs() < 1.0 + 1e-5);
            assert!(c[1].abs() <= 1.5 + 1e-5 && c[2].abs() <= 2.0 + 1e-5);
        }
    }

    #[test]
    fn without_provenance_it_declines_rather_than_guessing() {
        // Primitives self-tag; an untagged mesh is what a loader or a boolean
        // result looks like, and it must be left exactly as it is.
        let mut g = SphereGeometry::new(1.0, 8, 6);
        g.surfaces = None;
        assert!(retessellate(&g, 1e-3).is_none());
    }

    #[test]
    fn boundary_edge_count_distinguishes_open_from_closed() {
        // One triangle: three edges, each used once.
        assert_eq!(count_boundary_edges(&[0, 1, 2]), 3);
        // Two triangles sharing an edge: four boundary, one shared.
        assert_eq!(count_boundary_edges(&[0, 1, 2, 0, 2, 3]), 4);
        // A closed tetrahedron.
        assert_eq!(
            count_boundary_edges(&[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]),
            0
        );
    }

    #[test]
    fn reversed_winding_is_preserved_through_a_remesh() {
        let mut g = tagged_sphere(1.0, 16, 8);
        let table = g.surface_table().unwrap().clone();
        let flipped: Vec<u32> = g
            .index
            .as_ref()
            .unwrap()
            .chunks_exact(3)
            .flat_map(|t| [t[0], t[2], t[1]])
            .collect();
        g.set_index(flipped);
        g.set_surfaces(table);

        let (out, _) = retessellate(&g, 1e-2).unwrap();
        // Every triangle of an inside-out sphere winds clockwise seen from
        // outside, i.e. its face normal points at the centre.
        let pos = &out.get_attribute("position").unwrap().array;
        let idx = out.index.as_ref().unwrap();
        let mut inward = 0usize;
        let mut total = 0usize;
        for t in idx.chunks_exact(3) {
            let p: Vec<V3> = t
                .iter()
                .map(|&i| {
                    let o = i as usize * 3;
                    [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
                })
                .collect();
            let verts = [p[0], p[1], p[2]];
            if let (Some(fnorm), Some(c)) =
                (triangle_normal(&verts), v3::normalize(centroid(&verts)))
            {
                total += 1;
                if v3::dot(fnorm, c) < 0.0 {
                    inward += 1;
                }
            }
        }
        assert!(total > 0);
        assert!(
            inward * 10 > total * 9,
            "expected the reversed winding to survive: {inward}/{total} inward"
        );
    }
}
