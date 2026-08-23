//! What provenance buys immediately: analytic normals and parameter UVs.
//!
//! Neither needs topology, trimming, or a kernel change — only the surface,
//! which is why these are Stage 1 and not Stage 3.

use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::nurbs::v3;
use crate::nurbs::V3;

use super::table::{triangle_indices, triangle_normal, triangle_vertices};
use super::Surface;

/// Replace the `normal` attribute with the surface's analytic normal at each
/// vertex.
///
/// `compute_vertex_normals` averages the face normals of triangles that were
/// themselves an approximation — the error is bounded by the tessellation, so a
/// coarse mesh has visibly wrong shading even where its *positions* are exact.
/// This has no such error: the normal is evaluated on the surface itself.
///
/// # Winding
///
/// A [`Surface`] has one canonical outward sense, but a mesh may wind either
/// way against it (an inside-out shell, a subtracted cavity). The triangle's own
/// winding decides: if the face normal opposes the surface normal, the analytic
/// normal is flipped for that triangle's vertices. Anything else would silently
/// invert the shading on every cavity in a CSG result.
///
/// # Shared vertices
///
/// Where triangles from different surfaces share a vertex — a box corner — the
/// last write wins, which is wrong for exactly the vertices where a single
/// normal was already wrong. Those meshes want split vertices, which
/// [`mod@super::retessellate`] produces.
///
/// A vertex no triangle references is **left as found** — writing is per
/// triangle, so those are never reached. Returns the number of vertices written.
pub fn exact_normals(geometry: &mut BufferGeometry) -> Option<usize> {
    let table = geometry.surface_table()?.clone();
    let count = geometry.get_attribute("position")?.count();

    // Seed from the existing normals rather than from zeros.
    //
    // Writing is per *triangle*, so a vertex no triangle references is never
    // reached — and a UV sphere has two, at the ends of its pole rows. Starting
    // from zeros would silently replace their generator normal with `(0, 0, 0)`:
    // invisible, since nothing draws them, right up until something welds or
    // re-indexes the mesh and they become visible.
    let mut normals = match geometry.get_attribute("normal") {
        Some(a) if a.item_size == 3 && a.count() == count => a.array.clone(),
        _ => vec![0.0f32; count * 3],
    };
    let mut written = 0usize;

    for tri in 0..table.triangle_count() {
        let (Some(surface), Some(verts), Some(idx)) = (
            table.surface_of(tri),
            triangle_vertices(geometry, tri),
            triangle_indices(geometry, tri),
        ) else {
            continue;
        };

        let flip = match (
            triangle_normal(&verts),
            surface_normal_at(surface, verts[0]),
        ) {
            (Some(face), Some(sn)) => v3::dot(face, sn) < 0.0,
            _ => false,
        };

        for (k, &vi) in idx.iter().enumerate() {
            let Some(mut n) = surface_normal_at(surface, verts[k]) else {
                continue;
            };
            if flip {
                n = v3::scale(n, -1.0);
            }
            if vi * 3 + 2 < normals.len() {
                normals[vi * 3] = n[0] as f32;
                normals[vi * 3 + 1] = n[1] as f32;
                normals[vi * 3 + 2] = n[2] as f32;
                written += 1;
            }
        }
    }

    // `set_attribute` clears provenance, so put it back — the positions did not
    // change and the table still describes them.
    geometry.set_attribute("normal", BufferAttribute::new(normals, 3));
    geometry.set_surfaces(table);
    Some(written)
}

/// What [`exact_uvs`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UvReport {
    /// Vertices assigned a parameter.
    pub vertices: usize,
    /// Triangles that still cross a periodic seam after unwrapping.
    ///
    /// A vertex carries one uv. Where a mesh *shares* a vertex across the seam
    /// of a periodic surface — rather than duplicating it, as a UV sphere's
    /// wrap-around column does — no assignment can be right: the triangles
    /// either side need `u` and `u + 2π` from the same vertex. Those triangles
    /// are counted here rather than silently textured backwards. Splitting the
    /// seam vertices is the fix, and it changes the vertex buffer, so it is the
    /// caller's decision to make.
    pub seam_wrapped: usize,
}
/// Parameters for one patch, per **(triangle, corner)** rather than per vertex.
///
/// This is the distinction the whole seam problem turns on. A vertex carries one
/// uv, so a per-vertex assignment cannot even *express* the situation where two
/// triangles need `u` and `u + 2π` from the same vertex — it has to pick one and
/// call the other wrong. Assigning per corner can express it, which makes
/// splitting a mechanical consequence rather than a search.
struct PatchParams {
    /// `corner[t]` for the `t`-th triangle of the patch, or `None` where the
    /// surface could not be inverted at one of its vertices.
    corner: Vec<Option<[[f64; 2]; 3]>>,
    /// The patch's triangles, as global indices.
    triangles: Vec<usize>,
    periodic: (bool, bool),
}

/// Compute a globally consistent parameter for every corner of a patch.
///
/// Three steps, and each exists for a reason the previous one cannot handle:
///
/// 1. **Raw inversion** per vertex. `atan2` puts a branch cut somewhere on every
///    periodic surface, so these are inconsistent by construction.
/// 2. **Per-triangle unwrap.** Each triangle's three corners are shifted by whole
///    turns to sit near its own anchor, making every triangle individually
///    contiguous. A corner whose `u` is *arbitrary* — a sphere's pole, a cone's
///    apex — takes the mean of its neighbours instead of its meaningless raw
///    value, so it never drags a triangle across the cut.
/// 3. **Whole-turn offsets propagated across shared edges.** Breadth-first over
///    the triangle adjacency: a neighbour's offset is chosen so the two agree on
///    a vertex they share. This is what makes the assignment *global* rather than
///    per-triangle, and it is where the cut ends up — on the edges where
///    propagation closes a loop with a non-zero turn.
fn patch_parameters(
    geometry: &BufferGeometry,
    surface: &Surface,
    triangles: Vec<usize>,
) -> PatchParams {
    use std::collections::{HashSet, VecDeque};
    const TAU: f64 = std::f64::consts::TAU;

    let periodic = surface.periodic();

    let mut raw: HashMap<usize, (f64, f64)> = HashMap::new();
    let mut degenerate: HashSet<usize> = HashSet::new();
    let mut tri_idx: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
    let mut keep: Vec<usize> = Vec::with_capacity(triangles.len());

    for &tri in &triangles {
        let (Some(verts), Some(idx)) = (
            triangle_vertices(geometry, tri),
            triangle_indices(geometry, tri),
        ) else {
            continue;
        };
        for (k, &vi) in idx.iter().enumerate() {
            if let Some(p) = surface.invert(verts[k]) {
                raw.entry(vi).or_insert(p);
            }
            if surface.parameter_degenerate_at(verts[k]) {
                degenerate.insert(vi);
            }
        }
        tri_idx.push(idx);
        keep.push(tri);
    }

    // Step 2 — each triangle made contiguous on its own terms.
    let local: Vec<Option<[[f64; 2]; 3]>> = tri_idx
        .iter()
        .map(|idx| {
            let anchor = (0..3).find(|&k| !degenerate.contains(&idx[k]))?;
            let (au, av) = *raw.get(&idx[anchor])?;
            let mut out = [[0.0f64; 2]; 3];
            for k in 0..3 {
                let (u, v) = *raw.get(&idx[k])?;
                out[k] = [
                    if periodic.0 { unwrap_near(u, au) } else { u },
                    if periodic.1 { unwrap_near(v, av) } else { v },
                ];
            }
            // A degenerate corner's `u` is arbitrary; give it the mean of the
            // corners whose `u` means something, so it sits inside the triangle
            // instead of wherever `atan2` happened to put it.
            for k in 0..3 {
                if !degenerate.contains(&idx[k]) {
                    continue;
                }
                let others: Vec<f64> = (0..3)
                    .filter(|&j| j != k && !degenerate.contains(&idx[j]))
                    .map(|j| out[j][0])
                    .collect();
                if !others.is_empty() {
                    out[k][0] = others.iter().sum::<f64>() / others.len() as f64;
                }
            }
            Some(out)
        })
        .collect();

    // Step 3 — propagate whole-turn offsets across shared edges.
    let mut edge_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (t, idx) in tri_idx.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (idx[k], idx[(k + 1) % 3]);
            edge_map.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }

    let mut offset: Vec<Option<[f64; 2]>> = vec![None; tri_idx.len()];
    for seed in 0..tri_idx.len() {
        if offset[seed].is_some() || local[seed].is_none() {
            continue;
        }
        offset[seed] = Some([0.0, 0.0]);
        let mut queue = VecDeque::from([seed]);
        while let Some(t) = queue.pop_front() {
            let (Some(ot), Some(lt)) = (offset[t], local[t]) else {
                continue;
            };
            for k in 0..3 {
                let (a, b) = (tri_idx[t][k], tri_idx[t][(k + 1) % 3]);
                let Some(neighbours) = edge_map.get(&(a.min(b), a.max(b))) else {
                    continue;
                };
                for &t2 in neighbours {
                    if t2 == t || offset[t2].is_some() {
                        continue;
                    }
                    let Some(l2) = local[t2] else {
                        continue;
                    };
                    // Match on a vertex the two share.
                    let mut shift = [0.0f64; 2];
                    for &v in &[a, b] {
                        let (Some(k1), Some(k2)) = (
                            (0..3).find(|&i| tri_idx[t][i] == v),
                            (0..3).find(|&i| tri_idx[t2][i] == v),
                        ) else {
                            continue;
                        };
                        for (d, &per) in [periodic.0, periodic.1].iter().enumerate() {
                            if !per {
                                continue;
                            }
                            let want = lt[k1][d] + ot[d];
                            shift[d] = TAU * ((want - l2[k2][d]) / TAU).round();
                        }
                        break;
                    }
                    offset[t2] = Some(shift);
                    queue.push_back(t2);
                }
            }
        }
    }

    let corner = local
        .into_iter()
        .zip(&offset)
        .map(|(l, o)| {
            let (l, o) = (l?, (*o)?);
            let mut out = l;
            for c in out.iter_mut() {
                c[0] += o[0];
                c[1] += o[1];
            }
            Some(out)
        })
        .collect();

    PatchParams {
        corner,
        triangles: keep,
        periodic,
    }
}

/// Which value a vertex "wants", bucketed by whole turns, so two corners
/// disagreeing by exactly one turn are recognised as the same seam and not as
/// noise.
fn turn_bucket(value: f64, reference: f64, periodic: bool) -> i64 {
    if !periodic {
        return 0;
    }
    ((value - reference) / std::f64::consts::TAU).round() as i64
}

/// Every patch's per-corner parameters, computed once.
fn all_patch_parameters(
    geometry: &BufferGeometry,
) -> Option<(super::SurfaceTable, Vec<PatchParams>)> {
    let table = geometry.surface_table()?.clone();
    let patches = table
        .groups()
        .into_iter()
        .map(|(si, tris)| patch_parameters(geometry, &table.surfaces()[si], tris))
        .collect();
    Some((table, patches))
}

/// Corners that cannot get the value they want under a first-wins per-vertex
/// assignment — the seam, as a list.
///
/// `vertex_of(triangle, corner)` looks the corner's vertex up, so the same rule
/// can be applied before and after a split.
// The corner loops index a fixed-size `[_; 3]` in lockstep with a lookup keyed
// by the same `k`; zipping them reads worse than the subscript.
#[allow(clippy::needless_range_loop)]
fn seam_conflicts(
    patches: &[PatchParams],
    vertex_of: &dyn Fn(usize, usize) -> Option<usize>,
) -> (Vec<(usize, usize)>, usize) {
    let mut conflicts = Vec::new();
    let mut wrapped: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for patch in patches {
        let mut chosen: HashMap<usize, [f64; 2]> = HashMap::new();
        for (t, c) in patch.corner.iter().enumerate() {
            let Some(c) = c else { continue };
            for k in 0..3 {
                let Some(vi) = vertex_of(patch.triangles[t], k) else {
                    continue;
                };
                chosen.entry(vi).or_insert(c[k]);
            }
        }
        for (t, c) in patch.corner.iter().enumerate() {
            let Some(c) = c else { continue };
            for k in 0..3 {
                let Some(vi) = vertex_of(patch.triangles[t], k) else {
                    continue;
                };
                let Some(got) = chosen.get(&vi) else { continue };
                if turn_bucket(c[k][0], got[0], patch.periodic.0) != 0
                    || turn_bucket(c[k][1], got[1], patch.periodic.1) != 0
                {
                    conflicts.push((patch.triangles[t], k));
                    wrapped.insert(patch.triangles[t]);
                }
            }
        }
    }
    (conflicts, wrapped.len())
}

/// Write the `uv` attribute from per-corner parameters, normalized per patch.
fn write_uvs(
    geometry: &mut BufferGeometry,
    table: super::SurfaceTable,
    patches: &[PatchParams],
    seam_wrapped: usize,
) -> UvReport {
    let count = geometry
        .get_attribute("position")
        .map(|a| a.count())
        .unwrap_or(0);
    let mut uvs = match geometry.get_attribute("uv") {
        Some(a) if a.item_size == 2 && a.count() == count => a.array.clone(),
        _ => vec![0.0f32; count * 2],
    };
    let mut report = UvReport {
        vertices: 0,
        seam_wrapped,
    };

    for patch in patches {
        let (mut u0, mut u1) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY);
        for c in patch.corner.iter().flatten() {
            for p in c {
                u0 = u0.min(p[0]);
                u1 = u1.max(p[0]);
                v0 = v0.min(p[1]);
                v1 = v1.max(p[1]);
            }
        }
        if !u0.is_finite() {
            continue;
        }
        let du = if u1 - u0 > 1e-12 { u1 - u0 } else { 1.0 };
        let dv = if v1 - v0 > 1e-12 { v1 - v0 } else { 1.0 };

        for (t, c) in patch.corner.iter().enumerate() {
            let Some(c) = c else { continue };
            let Some(idx) = triangle_indices(geometry, patch.triangles[t]) else {
                continue;
            };
            for k in 0..3 {
                let vi = idx[k];
                if vi * 2 + 1 < uvs.len() {
                    uvs[vi * 2] = ((c[k][0] - u0) / du) as f32;
                    uvs[vi * 2 + 1] = ((c[k][1] - v0) / dv) as f32;
                    report.vertices += 1;
                }
            }
        }
    }

    geometry.set_attribute("uv", BufferAttribute::new(uvs, 2));
    geometry.set_surfaces(table);
    report
}

/// Replace the `uv` attribute with the surface's own parameters, normalized per
/// surface to `[0, 1]²` across the range its triangles actually occupy.
///
/// Normalizing per surface rather than globally is what makes this useful: each
/// face gets the full texture square, which is the layout a UV sphere's
/// hand-rolled mapping approximates and a box's per-face mapping does exactly.
///
/// # The seam
///
/// `atan2` has a branch cut, so a cylinder's `u` jumps by `2π` somewhere around
/// its circumference. Parameters are made globally consistent by unwrapping each
/// triangle and then propagating whole-turn offsets across shared edges — see
/// `patch_parameters`.
///
/// That still cannot help a vertex the seam runs *through*: the triangles either
/// side want `u` and `u + 2π` from it, and it holds one value.
/// [`UvReport::seam_wrapped`] counts those, and [`exact_uvs_split_seams`] fixes
/// them by giving the vertex a second copy.
///
/// As with the normals, a vertex no triangle references keeps whatever it had.
/// Only *assigned* parameters are guaranteed to lie in `[0, 1]²`.
pub fn exact_uvs(geometry: &mut BufferGeometry) -> Option<UvReport> {
    let (table, patches) = all_patch_parameters(geometry)?;
    let lookup = |tri: usize, k: usize| triangle_indices(geometry, tri).map(|i| i[k]);
    let (_, wrapped) = seam_conflicts(&patches, &lookup);
    Some(write_uvs(geometry, table, &patches, wrapped))
}

/// Cut the mesh along its parameter seams, then assign `uv`.
///
/// # What "cutting the seam" means
///
/// With parameters assigned per *corner* (`patch_parameters`), a seam is no
/// longer something to search for: it is exactly the set of corners whose value
/// disagrees, by whole turns, with the one their vertex ended up holding. Each
/// gets a private copy of that vertex, and is re-pointed at it.
///
/// # Why the parameters are computed before the split and not after
///
/// This is the whole difficulty. Splitting changes the mesh's connectivity —
/// a re-pointed corner no longer shares its edges — and the offsets in
/// `patch_parameters` propagate *along* that connectivity. Recomputing after
/// the split therefore produces a different assignment, with conflicts in new
/// places: a `16 × 10` UV sphere went 19 → 19, and an earlier
/// split-whole-triangles version converged to four instead of zero.
///
/// Computing once on the original mesh and then *reusing* those values makes the
/// split a pure re-indexing. Every corner ends up holding exactly the value it
/// asked for, so one pass suffices and nothing is left behind.
///
/// Positions, normals and every other attribute are copied to the new vertices,
/// so the mesh is geometrically identical — only the index buffer and the vertex
/// count change. The triangle *order* is untouched, so surface provenance still
/// describes it triangle-for-triangle.
pub fn exact_uvs_split_seams(geometry: &mut BufferGeometry) -> Option<UvReport> {
    let (table, patches) = all_patch_parameters(geometry)?;
    let lookup = |tri: usize, k: usize| triangle_indices(geometry, tri).map(|i| i[k]);
    let (conflicts, wrapped) = seam_conflicts(&patches, &lookup);
    if conflicts.is_empty() {
        return Some(write_uvs(geometry, table, &patches, wrapped));
    }

    let original_count = geometry.get_attribute("position")?.count();
    let mut index: Vec<u32> = match &geometry.index {
        Some(idx) => idx.clone(),
        None => (0..original_count as u32).collect(),
    };

    // One copy per conflicting corner. Two corners of the same vertex on the
    // same turn could share a copy; giving them one each costs a few vertices on
    // a seam and cannot be wrong, where sharing between different turns would be.
    let mut clones: Vec<usize> = Vec::new();
    for (tri, corner) in conflicts {
        let slot = tri * 3 + corner;
        if slot >= index.len() {
            continue;
        }
        clones.push(index[slot] as usize);
        index[slot] = (original_count + clones.len() - 1) as u32;
    }
    if clones.is_empty() {
        return Some(write_uvs(geometry, table, &patches, wrapped));
    }

    let names: Vec<String> = geometry.attributes.keys().cloned().collect();
    for name in names {
        let Some(attr) = geometry.attributes.get(&name) else {
            continue;
        };
        let item = attr.item_size;
        if attr.count() != original_count {
            continue;
        }
        let mut data = attr.array.clone();
        data.reserve(clones.len() * item);
        for &src in &clones {
            for c in 0..item {
                let v = data[src * item + c];
                data.push(v);
            }
        }
        geometry.set_attribute(name, BufferAttribute::new(data, item));
    }
    geometry.set_index(index);
    geometry.set_surfaces(table.clone());

    // Re-check with the *same* parameters against the new indices, so the report
    // is measured rather than assumed.
    let lookup = |tri: usize, k: usize| triangle_indices(geometry, tri).map(|i| i[k]);
    let (_, still) = seam_conflicts(&patches, &lookup);
    Some(write_uvs(geometry, table, &patches, still))
}

/// Shift `x` by whole turns to land nearest `anchor` — the fix for a triangle
/// whose vertices sit either side of `atan2`'s branch cut.
fn unwrap_near(x: f64, anchor: f64) -> f64 {
    use std::f64::consts::TAU;
    x - TAU * ((x - anchor) / TAU).round()
}

/// The surface normal at a *point*, by inverting to parameters first.
fn surface_normal_at(surface: &Surface, p: V3) -> Option<V3> {
    match surface {
        // A plane's normal is constant, so skip the inversion entirely.
        Surface::Plane { normal, .. } => Some(*normal),
        _ => {
            let (u, v) = surface.invert(p)?;
            surface.normal(u, v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::SphereGeometry;

    /// `SphereGeometry` tags itself now, and with the *generator's own* axis
    /// convention (Y-up, `φ = 0` at −X). Overriding that with a default Z-up
    /// sphere describes the same point set under a rotated parameterization, so
    /// the mesh's rows and columns stop aligning with it and triangles span
    /// arbitrary parameter ranges — which reads as a torn texture that is really
    /// a mismatched tag.
    fn attach_sphere(radius: f32, w: usize, h: usize) -> BufferGeometry {
        SphereGeometry::new(radius, w, h)
    }

    #[test]
    fn analytic_normals_beat_averaged_ones_on_a_coarse_sphere() {
        let coarse = attach_sphere(1.0, 8, 6);

        let mut averaged = coarse.clone();
        crate::compute_vertex_normals(&mut averaged);

        let mut exact = coarse.clone();
        assert!(exact_normals(&mut exact).unwrap() > 0);

        let err = |g: &BufferGeometry| {
            let pos = &g.get_attribute("position").unwrap().array;
            let nor = &g.get_attribute("normal").unwrap().array;
            let mut worst = 0.0f64;
            for (p, n) in pos.chunks_exact(3).zip(nor.chunks_exact(3)) {
                let pv = [p[0] as f64, p[1] as f64, p[2] as f64];
                let nv = [n[0] as f64, n[1] as f64, n[2] as f64];
                let (Some(r), Some(nn)) = (v3::normalize(pv), v3::normalize(nv)) else {
                    continue;
                };
                worst = worst.max(1.0 - v3::dot(r, nn));
            }
            worst
        };

        let e_avg = err(&averaged);
        let e_exact = err(&exact);
        assert!(
            e_avg > 1e-3,
            "expected averaging to be visibly wrong, got {e_avg}"
        );
        assert!(e_exact < 1e-9, "analytic normals still off by {e_exact}");
    }

    #[test]
    fn provenance_survives_writing_normals() {
        let mut g = attach_sphere(2.0, 12, 8);
        exact_normals(&mut g).unwrap();
        assert!(
            g.surface_table().is_some(),
            "writing an attribute must not cost the table that produced it"
        );
    }

    #[test]
    fn normals_follow_the_meshs_winding_not_the_surfaces() {
        // Reverse every triangle: the shell is now inside-out, and its normals
        // must invert with it or a subtracted cavity shades as a solid.
        let mut g = attach_sphere(1.0, 10, 6);
        let flipped_idx: Vec<u32> = g
            .index
            .as_ref()
            .unwrap()
            .chunks_exact(3)
            .flat_map(|t| [t[0], t[2], t[1]])
            .collect();
        let table = g.surface_table().unwrap().clone();
        g.set_index(flipped_idx);
        g.set_surfaces(table);

        exact_normals(&mut g).unwrap();
        let pos = &g.get_attribute("position").unwrap().array;
        let nor = &g.get_attribute("normal").unwrap().array;
        let mut inward = 0;
        for (p, n) in pos.chunks_exact(3).zip(nor.chunks_exact(3)) {
            let pv = [p[0] as f64, p[1] as f64, p[2] as f64];
            let nv = [n[0] as f64, n[1] as f64, n[2] as f64];
            if let (Some(r), Some(nn)) = (v3::normalize(pv), v3::normalize(nv)) {
                if v3::dot(r, nn) < -0.5 {
                    inward += 1;
                }
            }
        }
        assert!(inward > 0, "reversed winding produced no inward normals");
    }

    #[test]
    fn uvs_cover_the_unit_square_and_survive_the_seam() {
        let mut g = attach_sphere(1.0, 16, 10);
        let report = exact_uvs(&mut g).unwrap();
        assert!(report.vertices > 0);

        // Only vertices a triangle references are assigned; the rest keep the
        // generator's own values, which for a UV sphere sit slightly outside
        // [0, 1] by design (its pole columns carry a half-texel offset).
        let uv = &g.get_attribute("uv").unwrap().array;
        let referenced: std::collections::HashSet<usize> = g
            .index
            .as_ref()
            .unwrap()
            .iter()
            .map(|&i| i as usize)
            .collect();
        let (mut lo_u, mut hi_u, mut lo_v, mut hi_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for vi in referenced {
            let (u, v) = (uv[vi * 2], uv[vi * 2 + 1]);
            assert!(u.is_finite() && v.is_finite(), "non-finite uv");
            lo_u = lo_u.min(u);
            hi_u = hi_u.max(u);
            lo_v = lo_v.min(v);
            hi_v = hi_v.max(v);
        }
        assert!(lo_u >= -1e-6 && hi_u <= 1.0 + 1e-6, "u ∈ [{lo_u}, {hi_u}]");
        assert!(lo_v >= -1e-6 && hi_v <= 1.0 + 1e-6, "v ∈ [{lo_v}, {hi_v}]");
        assert!(
            hi_u - lo_u > 0.9 && hi_v - lo_v > 0.9,
            "uvs did not span the square"
        );
    }

    #[test]
    fn unwrapping_moves_a_value_by_whole_turns_only() {
        use std::f64::consts::{PI, TAU};
        let x = unwrap_near(-PI + 0.1, PI - 0.1);
        assert!((x - (PI + 0.1)).abs() < 1e-12, "got {x}");
        assert!(((x - (-PI + 0.1)) / TAU).fract().abs() < 1e-12);
    }

    #[test]
    fn splitting_seams_does_not_move_the_mesh() {
        let mut g = attach_sphere(2.0, 20, 12);
        exact_uvs_split_seams(&mut g).unwrap();

        // Every vertex is still exactly on the sphere, and the table still says so.
        let t = g.surface_table().unwrap();
        assert!(t.max_deviation(&g) < 1e-4, "the split moved geometry");
        for c in g.get_attribute("position").unwrap().array.chunks_exact(3) {
            let r = ((c[0] as f64).powi(2) + (c[1] as f64).powi(2) + (c[2] as f64).powi(2)).sqrt();
            assert!((r - 2.0).abs() < 1e-5, "radius {r}");
        }
    }

    /// Weld coincident vertices, collapsing a mesh's duplicated seam column into
    /// a shared one.
    ///
    /// Every built-in generator duplicates its seam, so none of them exercise
    /// the case [`exact_uvs_split_seams`] exists for. This manufactures it — and
    /// it is not artificial: a welded mesh is what comes back from a CSG result,
    /// an STL import, or any repair pass.
    fn weld(g: &BufferGeometry) -> BufferGeometry {
        use std::collections::HashMap;
        let pos = &g.get_attribute("position").unwrap().array;
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut remap: Vec<u32> = Vec::new();
        let mut kept: Vec<f32> = Vec::new();
        for c in pos.chunks_exact(3) {
            let key = [
                (c[0] as f64 * 1e6).round() as i64,
                (c[1] as f64 * 1e6).round() as i64,
                (c[2] as f64 * 1e6).round() as i64,
            ];
            let id = *map.entry(key).or_insert_with(|| {
                kept.extend_from_slice(c);
                (kept.len() / 3 - 1) as u32
            });
            remap.push(id);
        }
        let idx: Vec<u32> = g
            .index
            .as_ref()
            .unwrap()
            .iter()
            .map(|&i| remap[i as usize])
            .collect();

        let table = g.surface_table().unwrap().clone();
        let mut out = BufferGeometry::new();
        out.set_attribute("position", BufferAttribute::new(kept, 3));
        out.set_index(idx);
        out.set_surfaces(table);
        out
    }

    /// Worst triangle span in a *periodic* direction, which is the only place a
    /// texture can tear.
    ///
    /// Spanning the full range in a non-periodic direction is ordinary: a cone
    /// built with one height segment has every side triangle running apex to
    /// rim, so `v` spans 1.0 by construction and nothing is wrong.
    fn worst_periodic_span(g: &BufferGeometry) -> f32 {
        let table = g.surface_table().unwrap();
        let uv = &g.get_attribute("uv").unwrap().array;
        let mut worst = 0.0f32;
        for (si, tris) in table.groups() {
            let (pu, pv) = table.surfaces()[si].periodic();
            if !pu && !pv {
                continue;
            }
            for tri in tris {
                let Some(idx) = crate::brep::triangle_indices(g, tri) else {
                    continue;
                };
                let (mut lo_u, mut hi_u) = (f32::MAX, f32::MIN);
                let (mut lo_v, mut hi_v) = (f32::MAX, f32::MIN);
                for &i in &idx {
                    let (u, v) = (uv[i * 2], uv[i * 2 + 1]);
                    lo_u = lo_u.min(u);
                    hi_u = hi_u.max(u);
                    lo_v = lo_v.min(v);
                    hi_v = hi_v.max(v);
                }
                if pu {
                    worst = worst.max(hi_u - lo_u);
                }
                if pv {
                    worst = worst.max(hi_v - lo_v);
                }
            }
        }
        worst
    }

    #[test]
    fn a_welded_seam_tears_and_the_split_repairs_it() {
        // The case the function exists for, on the meshes it exists for.
        use crate::geometries::{CylinderGeometry, TorusGeometry};
        const TAU: f32 = std::f32::consts::PI * 2.0;

        let cases: Vec<(&str, BufferGeometry)> = vec![
            ("sphere", SphereGeometry::new(1.0, 20, 12)),
            (
                "cylinder",
                CylinderGeometry::new(1.0, 1.0, 2.0, 20, 2, true, 0.0, TAU),
            ),
            ("torus", TorusGeometry::new(3.0, 1.0, 12, 20, TAU)),
        ];

        for (name, g) in cases {
            let welded = weld(&g);

            let mut torn = welded.clone();
            let before = exact_uvs(&mut torn).unwrap();
            assert!(
                before.seam_wrapped > 0,
                "{name}: welding did not produce a shared seam"
            );
            assert!(
                worst_periodic_span(&torn) > 0.5,
                "{name}: expected a visibly torn texture before the split"
            );

            let mut fixed = welded.clone();
            let after = exact_uvs_split_seams(&mut fixed).unwrap();
            assert_eq!(
                after.seam_wrapped, 0,
                "{name}: {} wrapped triangles left (was {})",
                after.seam_wrapped, before.seam_wrapped
            );
            let span = worst_periodic_span(&fixed);
            assert!(
                span < 0.5,
                "{name}: a triangle still spans {span} of the texture"
            );
            assert!(
                fixed.surface_table().unwrap().max_deviation(&fixed) < 1e-4,
                "{name}: the split moved geometry"
            );
        }
    }

    #[test]
    fn the_built_in_generators_need_no_split() {
        // They all duplicate their seam column already, so the split is a no-op —
        // worth asserting, because a split that fired here would be adding
        // vertices for nothing.
        use crate::geometries::{CylinderGeometry, TorusGeometry};
        const TAU: f32 = std::f32::consts::PI * 2.0;
        let cases: Vec<(&str, BufferGeometry)> = vec![
            ("sphere", SphereGeometry::new(1.0, 24, 14)),
            ("sphere/coarse", SphereGeometry::new(1.0, 6, 4)),
            (
                "capped cylinder",
                CylinderGeometry::new(1.0, 1.0, 2.0, 20, 2, false, 0.0, TAU),
            ),
            (
                "cone",
                CylinderGeometry::new(0.0, 1.5, 3.0, 18, 1, false, 0.0, TAU),
            ),
            ("torus", TorusGeometry::new(3.0, 1.0, 12, 20, TAU)),
        ];
        for (name, mut g) in cases {
            let before = g.get_attribute("position").unwrap().count();
            let r = exact_uvs_split_seams(&mut g).unwrap();
            assert_eq!(r.seam_wrapped, 0, "{name}");
            assert_eq!(
                g.get_attribute("position").unwrap().count(),
                before,
                "{name}: vertices were duplicated with nothing to fix"
            );
            assert!(worst_periodic_span(&g) < 0.5, "{name}: torn without a weld");
        }
    }

    #[test]
    fn splitting_is_a_pure_reindexing() {
        // Every new vertex is a copy of an existing one: the set of positions is
        // unchanged, only the count and the index buffer.
        let mut g = attach_sphere(1.0, 16, 10);
        let before: Vec<[f32; 3]> = g
            .get_attribute("position")
            .unwrap()
            .array
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let tris_before = crate::brep::triangle_count(&g);

        exact_uvs_split_seams(&mut g).unwrap();

        assert_eq!(crate::brep::triangle_count(&g), tris_before);
        let after: Vec<[f32; 3]> = g
            .get_attribute("position")
            .unwrap()
            .array
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        assert_eq!(
            &after[..before.len()],
            &before[..],
            "existing vertices moved"
        );
        for p in &after[before.len()..] {
            assert!(
                before.contains(p),
                "a new vertex is not a copy of an old one"
            );
        }
    }

    #[test]
    fn a_poles_arbitrary_parameter_is_not_mistaken_for_a_seam_crossing() {
        // Every triangle around a sphere's pole touches a vertex whose `u` is
        // meaningless. Counting those as seam crossings reported six on a mesh
        // that has two, and would have split four triangles for nothing.
        let s = Surface::sphere([0.0; 3], 1.0).with_axis([0.0, 1.0, 0.0]);
        assert!(s.parameter_degenerate_at([0.0, 1.0, 0.0]), "north pole");
        assert!(s.parameter_degenerate_at([0.0, -1.0, 0.0]), "south pole");
        assert!(
            !s.parameter_degenerate_at([1.0, 0.0, 0.0]),
            "the equator is fine"
        );

        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        assert!(c.parameter_degenerate_at([0.0, 0.0, 5.0]), "the axis");
        assert!(!c.parameter_degenerate_at([2.0, 0.0, 0.0]));

        // A plane and a torus have no such place.
        assert!(!Surface::plane([0.0; 3], [0.0, 0.0, 1.0]).parameter_degenerate_at([0.0; 3]));
        assert!(
            !Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0).parameter_degenerate_at([0.0; 3])
        );
    }

    #[test]
    fn without_provenance_nothing_happens() {
        // Primitives self-tag now, so drop the table to get the untagged case —
        // which is still every mesh that arrives from a loader or a CSG result.
        let mut g = SphereGeometry::new(1.0, 8, 6);
        g.surfaces = None;
        assert!(exact_normals(&mut g).is_none());
        assert!(exact_uvs(&mut g).is_none());
    }
}

// -----------------------------------------------------------------------------
// Recovering provenance from a finished mesh
// -----------------------------------------------------------------------------
// Provenance is normally carried FROM the primitives, THROUGH the booleans. When
// it is not -- and the arrangement kernel does not carry it -- everything
// downstream that needs the analytic identity of a face is stranded: STEP export
// wants a cylinder with a parameter range, not a band of triangles, and
// `Body::from_tagged_mesh` cannot find an edge without knowing which two
// surfaces disagree across it.
//
// This recovers what can be recovered by fitting. It is not a substitute for
// carrying provenance properly: a fitted plane is a plane the triangles happen
// to lie in, not the plane the model was written with, and the two differ
// wherever a boolean has cut one face into pieces that are no longer one face.
// What it does give is a table where there was none, for the large majority of
// this kind of geometry, which is flat.
//
// PLANES ONLY, deliberately. Cylinders need axis fitting and a coaxiality test
// across a patch, and a wrong cylinder is worse than an honest plane: it would
// export a curved face with a radius nobody chose. Triangles that are not part
// of a planar patch of at least `min_patch` get their own single-triangle plane,
// which is exact, and leaves a tessellated cylinder looking like what it is.

use super::table::SurfaceTable;

/// How the fit went.
#[derive(Debug, Clone, Copy)]
pub struct FitReport {
    /// Distinct planar patches found.
    pub patches: usize,
    /// Triangles that joined a patch of at least `min_patch`.
    pub merged: usize,
    /// Triangles left as their own plane.
    pub singletons: usize,
}

/// Fit planes to a triangle soup and attach a [`SurfaceTable`].
///
/// Groups triangles by their plane -- normal and offset, quantised by `tol` --
/// and gives each group one `Surface::Plane`. Returns `None` if the geometry has
/// no positions to work from.
pub fn recover_planar_surfaces(
    geometry: &mut BufferGeometry,
    tol: f64,
    min_patch: usize,
) -> Option<FitReport> {
    let n_tris = geometry.draw_count() / 3;
    if n_tris == 0 {
        return None;
    }
    let mut key_of: Vec<[i64; 4]> = Vec::with_capacity(n_tris);
    for t in 0..n_tris {
        let v = triangle_vertices(geometry, t)?;
        let n = triangle_normal(&v)?;
        let d = n[0] * v[0][0] + n[1] * v[0][1] + n[2] * v[0][2];
        // Quantise the plane, not the triangle. Two coplanar triangles must land
        // on the same key or the patch splits; the sign is normalised so a face
        // and its opposite do not merge.
        let q = |x: f64| (x / tol).round() as i64;
        key_of.push([q(n[0]), q(n[1]), q(n[2]), q(d)]);
    }
    let mut counts: HashMap<[i64; 4], usize> = HashMap::new();
    for k in &key_of {
        *counts.entry(*k).or_insert(0) += 1;
    }
    let mut index: HashMap<[i64; 4], usize> = HashMap::new();
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut tri_face: Vec<u32> = Vec::with_capacity(n_tris);
    let (mut merged, mut singletons) = (0usize, 0usize);
    for t in 0..n_tris {
        let v = triangle_vertices(geometry, t)?;
        let n = triangle_normal(&v)?;
        let big = counts[&key_of[t]] >= min_patch;
        let slot = if big {
            merged += 1;
            let k = key_of[t];
            if let Some(&i) = index.get(&k) {
                i
            } else {
                let i = surfaces.len();
                surfaces.push(Surface::Plane {
                    origin: v[0],
                    normal: n,
                    x_dir: any_perp(n),
                });
                index.insert(k, i);
                i
            }
        } else {
            singletons += 1;
            let i = surfaces.len();
            surfaces.push(Surface::Plane {
                origin: v[0],
                normal: n,
                x_dir: any_perp(n),
            });
            i
        };
        tri_face.push(slot as u32);
    }
    let patches = index.len();
    let table = SurfaceTable::new(surfaces, tri_face)?;
    geometry.set_surfaces(table);
    Some(FitReport {
        patches,
        merged,
        singletons,
    })
}

/// Fit planes AND cylinders. Same contract as [`recover_planar_surfaces`], but
/// triangles left over from the planar pass get one more chance: grouped into
/// connected patches and tested against an axis-aligned cylinder.
///
/// Axis-aligned only. A general axis needs the normals' null space and a
/// least-squares line, and getting it slightly wrong exports a cylinder whose
/// axis is not the one the model was built on -- which is worse than the honest
/// per-triangle planes it would replace. X, Y and Z cover geometry written the
/// way OpenSCAD writes it.
pub fn recover_surfaces(
    geometry: &mut BufferGeometry,
    tol: f64,
    min_patch: usize,
) -> Option<FitReport> {
    let n_tris = geometry.draw_count() / 3;
    if n_tris == 0 {
        return None;
    }
    // --- planar pass, as before
    let mut key_of: Vec<[i64; 4]> = Vec::with_capacity(n_tris);
    let mut norm_of: Vec<V3> = Vec::with_capacity(n_tris);
    let mut vert_of: Vec<[V3; 3]> = Vec::with_capacity(n_tris);
    for t in 0..n_tris {
        let v = triangle_vertices(geometry, t)?;
        let n = triangle_normal(&v)?;
        let d = n[0] * v[0][0] + n[1] * v[0][1] + n[2] * v[0][2];
        let q = |x: f64| (x / tol).round() as i64;
        key_of.push([q(n[0]), q(n[1]), q(n[2]), q(d)]);
        norm_of.push(n);
        vert_of.push(v);
    }
    let mut counts: HashMap<[i64; 4], usize> = HashMap::new();
    for k in &key_of {
        *counts.entry(*k).or_insert(0) += 1;
    }
    // --- group the leftovers by shared vertices
    let quantum = tol.max(1e-6);
    let qp = |p: V3| {
        [
            (p[0] / quantum).round() as i64,
            (p[1] / quantum).round() as i64,
            (p[2] / quantum).round() as i64,
        ]
    };
    let mut owner: Vec<usize> = (0..n_tris).collect();
    fn find(o: &mut [usize], mut i: usize) -> usize {
        while o[i] != i {
            o[i] = o[o[i]];
            i = o[i];
        }
        i
    }
    // Shared vertex AND a small dihedral. Vertex alone was the limit: in the
    // wing every hinge body touches its neighbours, so the entire leftover set
    // collapsed into one component, which is not a cylinder, and 25 363 triangles
    // were rejected together. A tessellated barrel turns ~18 degrees per facet
    // ($fn 20); where it meets its pad or its cap the turn is far larger, so a
    // 40-degree cut separates the curved patch from everything it abuts.
    const SMOOTH: f64 = 0.766; // cos 40 deg
    let mut seen: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for t in 0..n_tris {
        if counts[&key_of[t]] >= min_patch {
            continue;
        }
        for v in &vert_of[t] {
            let e = seen.entry(qp(*v)).or_default();
            for &u in e.iter() {
                let (a, b) = (norm_of[t], norm_of[u]);
                if a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < SMOOTH {
                    continue;
                }
                let (ra, rb) = (find(&mut owner, t), find(&mut owner, u));
                if ra != rb {
                    owner[ra] = rb;
                }
            }
            e.push(t);
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for t in 0..n_tris {
        if counts[&key_of[t]] >= min_patch {
            continue;
        }
        let r = find(&mut owner, t);
        groups.entry(r).or_default().push(t);
    }
    // --- try a cylinder on each group
    let mut cyl_of: HashMap<usize, usize> = HashMap::new();
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut cylinders = 0usize;
    for (root, members) in &groups {
        if members.len() < 6 {
            continue;
        }
        // GENERAL AXIS, not X/Y/Z. Axis-aligned fitting recovered almost
        // nothing on the wing -- 26 160 singletons down to 25 363 -- because the
        // panels are rotated when stowed, so every hinge barrel on them lies on a
        // skew axis. The restriction was the limit, not the geometry.
        //
        // A cylinder's face normals are all perpendicular to its axis, so the
        // axis is the direction that minimises sum (n . d)^2 -- the smallest
        // eigenvector of the normal covariance. Power-iterate on tr(M)I - M,
        // whose LARGEST eigenvector is that direction, which avoids needing an
        // eigen decomposition at all.
        {
            let mut m = [[0.0f64; 3]; 3];
            for &t in members {
                let n = norm_of[t];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] += n[i] * n[j];
                    }
                }
            }
            let tr = m[0][0] + m[1][1] + m[2][2];
            let mut d = [0.5773, 0.5774, 0.5775];
            for _ in 0..64 {
                let mut w2 = [0.0f64; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        w2[i] += ((if i == j { tr } else { 0.0 }) - m[i][j]) * d[j];
                    }
                }
                let l = (w2[0] * w2[0] + w2[1] * w2[1] + w2[2] * w2[2]).sqrt();
                if l < 1e-12 {
                    break;
                }
                d = [w2[0] / l, w2[1] / l, w2[2] / l];
            }
            // every normal must actually be perpendicular to it
            if members.iter().all(|&t| {
                (norm_of[t][0] * d[0] + norm_of[t][1] * d[1] + norm_of[t][2] * d[2]).abs() < 0.12
            }) {
                let ua = any_perp(d);
                let wa = [
                    d[1] * ua[2] - d[2] * ua[1],
                    d[2] * ua[0] - d[0] * ua[2],
                    d[0] * ua[1] - d[1] * ua[0],
                ];
                let pr = |v: V3| {
                    (
                        v[0] * ua[0] + v[1] * ua[1] + v[2] * ua[2],
                        v[0] * wa[0] + v[1] * wa[1] + v[2] * wa[2],
                    )
                };
                let (
                    mut sx,
                    mut sy,
                    mut sxx,
                    mut syy,
                    mut sxy,
                    mut sx3,
                    mut sy3,
                    mut sxy2,
                    mut sx2y,
                    mut n,
                ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for &t in members {
                    for v in &vert_of[t] {
                        let (x, y) = pr(*v);
                        sx += x;
                        sy += y;
                        sxx += x * x;
                        syy += y * y;
                        sxy += x * y;
                        sx3 += x * x * x;
                        sy3 += y * y * y;
                        sxy2 += x * y * y;
                        sx2y += x * x * y;
                        n += 1.0;
                    }
                }
                let (c1, c2) = (sxx - sx * sx / n, sxy - sx * sy / n);
                let c3 = syy - sy * sy / n;
                let d1 = 0.5 * (sx3 + sxy2 - (sx * (sxx + syy)) / n);
                let d2 = 0.5 * (sy3 + sx2y - (sy * (sxx + syy)) / n);
                let det = c1 * c3 - c2 * c2;
                if det.abs() > 1e-9 {
                    let cu = (d1 * c3 - d2 * c2) / det;
                    let cw = (d2 * c1 - d1 * c2) / det;
                    let mut r2 = 0.0;
                    for &t in members {
                        for v in &vert_of[t] {
                            let (x, y) = pr(*v);
                            r2 += (x - cu).powi(2) + (y - cw).powi(2);
                        }
                    }
                    let r = (r2 / n).sqrt();
                    let worst = members
                        .iter()
                        .flat_map(|&t| vert_of[t])
                        .map(|v| {
                            let (x, y) = pr(v);
                            (((x - cu).powi(2) + (y - cw).powi(2)).sqrt() - r).abs()
                        })
                        .fold(0.0f64, f64::max);
                    if r > 0.05 && worst <= (0.02 * r).max(tol * 10.0) {
                        let origin = [
                            cu * ua[0] + cw * wa[0],
                            cu * ua[1] + cw * wa[1],
                            cu * ua[2] + cw * wa[2],
                        ];
                        cyl_of.insert(*root, surfaces.len());
                        surfaces.push(Surface::Cylinder {
                            origin,
                            axis: d,
                            x_dir: ua,
                            radius: r,
                        });
                        cylinders += 1;
                        continue;
                    }
                }
            }
        }
        for ax in 0..3usize {
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            if members.iter().any(|&t| norm_of[t][ax].abs() > 0.08) {
                continue;
            }
            // algebraic circle fit on the projected vertices
            let (
                mut sx,
                mut sy,
                mut sxx,
                mut syy,
                mut sxy,
                mut sx3,
                mut sy3,
                mut sxy2,
                mut sx2y,
                mut n,
            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for &t in members {
                for v in &vert_of[t] {
                    let (x, y) = (v[u], v[w]);
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                    sx3 += x * x * x;
                    sy3 += y * y * y;
                    sxy2 += x * y * y;
                    sx2y += x * x * y;
                    n += 1.0;
                }
            }
            let (c1, c2) = (sxx - sx * sx / n, sxy - sx * sy / n);
            let c3 = syy - sy * sy / n;
            let d1 = 0.5 * (sx3 + sxy2 - (sx * (sxx + syy)) / n);
            let d2 = 0.5 * (sy3 + sx2y - (sy * (sxx + syy)) / n);
            let det = c1 * c3 - c2 * c2;
            if det.abs() < 1e-9 {
                continue;
            }
            let cu = (d1 * c3 - d2 * c2) / det;
            let cw = (d2 * c1 - d1 * c2) / det;
            let mut r2 = 0.0;
            for &t in members {
                for v in &vert_of[t] {
                    r2 += (v[u] - cu).powi(2) + (v[w] - cw).powi(2);
                }
            }
            let r = (r2 / n).sqrt();
            if r < 0.05 {
                continue;
            }
            let worst = members
                .iter()
                .flat_map(|&t| vert_of[t])
                .map(|v| (((v[u] - cu).powi(2) + (v[w] - cw).powi(2)).sqrt() - r).abs())
                .fold(0.0f64, f64::max);
            if worst > (0.02 * r).max(tol * 10.0) {
                continue;
            }
            let mut origin = [0.0; 3];
            origin[u] = cu;
            origin[w] = cw;
            let mut axis = [0.0; 3];
            axis[ax] = 1.0;
            let mut x_dir = [0.0; 3];
            x_dir[u] = 1.0;
            cyl_of.insert(*root, surfaces.len());
            surfaces.push(Surface::Cylinder {
                origin,
                axis,
                x_dir,
                radius: r,
            });
            cylinders += 1;
            break;
        }
    }
    // --- assemble
    let mut index: HashMap<[i64; 4], usize> = HashMap::new();
    let mut tri_face: Vec<u32> = Vec::with_capacity(n_tris);
    let (mut merged, mut singletons) = (0usize, 0usize);
    for t in 0..n_tris {
        let slot = if counts[&key_of[t]] >= min_patch {
            merged += 1;
            let k = key_of[t];
            *index.entry(k).or_insert_with(|| {
                surfaces.push(Surface::Plane {
                    origin: vert_of[t][0],
                    normal: norm_of[t],
                    x_dir: any_perp(norm_of[t]),
                });
                surfaces.len() - 1
            })
        } else {
            let r = find(&mut owner, t);
            match cyl_of.get(&r) {
                Some(&i) => {
                    merged += 1;
                    i
                }
                None => {
                    singletons += 1;
                    surfaces.push(Surface::Plane {
                        origin: vert_of[t][0],
                        normal: norm_of[t],
                        x_dir: any_perp(norm_of[t]),
                    });
                    surfaces.len() - 1
                }
            }
        };
        tri_face.push(slot as u32);
    }
    let patches = index.len() + cylinders;
    let table = SurfaceTable::new(surfaces, tri_face)?;
    geometry.set_surfaces(table);
    Some(FitReport {
        patches,
        merged,
        singletons,
    })
}

fn any_perp(n: V3) -> V3 {
    let t = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let c = [
        t[1] * n[2] - t[2] * n[1],
        t[2] * n[0] - t[0] * n[2],
        t[0] * n[1] - t[1] * n[0],
    ];
    let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    [c[0] / l, c[1] / l, c[2] / l]
}
