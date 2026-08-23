//! Curvature-adaptive tessellation to `BufferGeometry`.
//!
//! One tessellator, driven by a chord tolerance, replacing the per-primitive
//! hand-rolled meshers that each pick their own segment counts. Two properties
//! make it worth the indirection:
//!
//! * **Normals are analytic.** They come from `Sᵤ × Sᵥ`, not from averaging the
//!   face normals of the triangles we just made up. On a sphere the difference
//!   is the whole faceting artifact.
//! * **Sample density follows curvature.** A flat region gets two samples and a
//!   tight fillet gets as many as the tolerance demands, instead of a uniform
//!   grid sized for the worst spot on the surface.
//!
//! This is also the piece that makes surface provenance (Stage 1) pay: with the
//! surface retained, a boolean result can be re-tessellated at render tolerance
//! instead of being stuck with whatever `$fn` was chosen before the boolean ran.

use crate::core::{BufferAttribute, BufferGeometry};

use super::curve::NurbsCurve;
use super::surface::NurbsSurface;
use super::v3;
use super::V3;

/// How finely to sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TessellationOptions {
    /// Maximum distance between the true surface and the mesh chord, in model
    /// units. The dominant control.
    pub tolerance: f64,
    /// Hard floor on samples per direction, applied after adaptation. Keeps a
    /// flat patch from degenerating to a single quad when a caller wants a grid.
    pub min_samples: usize,
    /// Ceiling on samples per direction.
    ///
    /// Enforced by bounding the recursion depth, not by decimating a finished
    /// sample set — decimation would silently break a tolerance the sampler had
    /// already achieved. When this ceiling binds, the chord tolerance is **not**
    /// met; [`sample_grid_meets_tolerance`] reports whether that happened rather
    /// than leaving the caller to assume it did not.
    ///
    /// The surface's own breakpoint count is a floor this cannot push below:
    /// dropping a knot line would smooth over a crease the surface really has,
    /// which is a wrong mesh rather than a coarse one. A four-span direction
    /// therefore yields at least five samples however small this is set.
    pub max_samples: usize,
    /// Emit a `uv` attribute (normalized surface parameters).
    pub uvs: bool,
}

impl Default for TessellationOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-3,
            min_samples: 2,
            max_samples: 1024,
            uvs: true,
        }
    }
}

impl TessellationOptions {
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            ..Self::default()
        }
    }
}

/// Worst deviation of the true geometry from the chord over `[a, b]`, sampled
/// at the quarter, half and three-quarter points.
///
/// The half point alone is the textbook sagitta and is the chord error to first
/// order — but it is blind to a span that returns to its own chord midpoint,
/// which a cubic-or-higher loop can do. Three probes cost three evaluations and
/// close that hole.
///
/// `at` returns the geometry's position for a parameter; for a surface it
/// returns the *worst* position across a set of probe iso-lines, so a direction
/// is refined for whichever iso-line curves the most.
fn chord_deviation(a: f64, b: f64, at: &mut dyn FnMut(f64) -> Vec<V3>) -> f64 {
    let pa = at(a);
    let pb = at(b);
    let mut worst = 0.0f64;
    for k in 1..4 {
        let t = k as f64 / 4.0;
        let m = a + (b - a) * t;
        let pm = at(m);
        for ((qa, qb), qm) in pa.iter().zip(pb.iter()).zip(pm.iter()) {
            let chord = v3::add(v3::scale(*qa, 1.0 - t), v3::scale(*qb, t));
            worst = worst.max(v3::dist(*qm, chord));
        }
    }
    worst
}

fn adaptive_split(
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
    max_depth: usize,
    at: &mut dyn FnMut(f64) -> Vec<V3>,
    out: &mut Vec<f64>,
) {
    if depth >= max_depth || chord_deviation(a, b, at) <= tol {
        return;
    }
    let m = 0.5 * (a + b);
    adaptive_split(a, m, tol, depth + 1, max_depth, at, out);
    out.push(m);
    adaptive_split(m, b, tol, depth + 1, max_depth, at, out);
}

/// Recursion depth that keeps the total sample count within `max_samples`.
///
/// Bisection produces up to `2^depth` intervals per breakpoint interval, so the
/// budget divides across them.
fn depth_budget(n_intervals: usize, max_samples: usize) -> usize {
    let per_interval = (max_samples.saturating_sub(1)) / n_intervals.max(1);
    if per_interval < 2 {
        return 0;
    }
    (per_interval as f64).log2().floor() as usize
}

/// Build the sample parameters for one direction: every breakpoint, plus
/// whatever adaptive subdivision the tolerance demands between them.
///
/// Breakpoints are always sampled because curvature is generally discontinuous
/// across a knot — subdividing blindly through one would smooth over a crease
/// the surface actually has.
fn sample_params(
    breaks: &[f64],
    opts: &TessellationOptions,
    at: &mut dyn FnMut(f64) -> Vec<V3>,
) -> Vec<f64> {
    let max_depth = depth_budget(breaks.len() - 1, opts.max_samples);

    let mut params = vec![breaks[0]];
    for w in breaks.windows(2) {
        let mut interior = Vec::new();
        adaptive_split(w[0], w[1], opts.tolerance, 0, max_depth, at, &mut interior);
        params.extend(interior);
        params.push(w[1]);
    }

    // Enforce the floor by uniform bisection. Capped by `max_samples` so a
    // caller setting min > max gets the max rather than an argument error.
    let floor = opts.min_samples.max(2).min(opts.max_samples.max(2));
    while params.len() < floor {
        let mut refined = Vec::with_capacity(params.len() * 2 - 1);
        for w in params.windows(2) {
            refined.push(w[0]);
            refined.push(0.5 * (w[0] + w[1]));
        }
        refined.push(*params.last().unwrap());
        params = refined;
    }
    params
}

/// Tessellate a curve to a polyline honouring `tolerance`.
pub fn tessellate_curve(curve: &NurbsCurve, opts: &TessellationOptions) -> Vec<V3> {
    let breaks = curve.breakpoints();
    let mut at = |u: f64| vec![curve.point(u)];
    let params = sample_params(&breaks, opts, &mut at);
    params.into_iter().map(|u| curve.point(u)).collect()
}

/// Breakpoints plus their midpoints — the iso-lines each direction is probed
/// against.
fn probe_lines(breaks: &[f64]) -> Vec<f64> {
    let mut p: Vec<f64> = breaks.to_vec();
    p.extend(breaks.windows(2).map(|w| 0.5 * (w[0] + w[1])));
    p.sort_by(|x, y| x.partial_cmp(y).unwrap());
    p
}

/// The `(u, v)` sample grid a surface needs at this tolerance.
///
/// Deviation in `u` is measured along several iso-`v` lines and vice versa, so a
/// surface that is flat along one iso-line and curved along the next is refined
/// for the curved one. Probing *every* iso-line would be exact and quadratic;
/// probing the breakpoints plus their midpoints catches what arises in practice
/// at a fraction of the cost.
pub fn sample_grid(surface: &NurbsSurface, opts: &TessellationOptions) -> (Vec<f64>, Vec<f64>) {
    let bu = surface.breakpoints_u();
    let bv = surface.breakpoints_v();
    let probe_v = probe_lines(&bv);
    let probe_u = probe_lines(&bu);

    let mut at_u = |u: f64| probe_v.iter().map(|&v| surface.point(u, v)).collect();
    let params_u = sample_params(&bu, opts, &mut at_u);

    let mut at_v = |v: f64| probe_u.iter().map(|&u| surface.point(u, v)).collect();
    let params_v = sample_params(&bv, opts, &mut at_v);

    (params_u, params_v)
}

/// Did the sample grid actually achieve the requested tolerance, or did
/// `max_samples` bind first?
///
/// The plan's "no silent caps" rule: a truncated sampling reads as "covered
/// everything" unless the caller can ask. Re-measures the chord deviation on
/// the produced grid rather than inferring it from the sample counts.
pub fn sample_grid_meets_tolerance(
    surface: &NurbsSurface,
    params_u: &[f64],
    params_v: &[f64],
    tolerance: f64,
) -> bool {
    let probe_v = probe_lines(&surface.breakpoints_v());
    let probe_u = probe_lines(&surface.breakpoints_u());

    let mut at_u = |u: f64| probe_v.iter().map(|&v| surface.point(u, v)).collect();
    if params_u
        .windows(2)
        .any(|w| chord_deviation(w[0], w[1], &mut at_u) > tolerance)
    {
        return false;
    }

    let mut at_v = |v: f64| probe_u.iter().map(|&u| surface.point(u, v)).collect();
    !params_v
        .windows(2)
        .any(|w| chord_deviation(w[0], w[1], &mut at_v) > tolerance)
}

/// Tessellate a surface to an indexed `BufferGeometry` with `position`,
/// `normal` and (optionally) `uv`.
///
/// Degenerate triangles — the ones a collapsed control row produces at a pole —
/// are dropped rather than emitted with a zero-length normal, the same thing
/// `SphereGeometry` does at its poles but derived from the geometry instead of
/// hard-coded per primitive.
pub fn tessellate_surface(surface: &NurbsSurface, opts: &TessellationOptions) -> BufferGeometry {
    let (params_u, params_v) = sample_grid(surface, opts);
    tessellate_surface_grid(surface, &params_u, &params_v, opts)
}

/// Tessellate on an explicit parameter grid — the escape hatch for callers that
/// need a specific vertex count (parity tests, fixed-resolution exports).
pub fn tessellate_surface_grid(
    surface: &NurbsSurface,
    params_u: &[f64],
    params_v: &[f64],
    opts: &TessellationOptions,
) -> BufferGeometry {
    let (nu, nv) = (params_u.len(), params_v.len());
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let span_u = (u1 - u0).max(f64::MIN_POSITIVE);
    let span_v = (v1 - v0).max(f64::MIN_POSITIVE);

    let mut positions = Vec::with_capacity(nu * nv * 3);
    let mut normals = Vec::with_capacity(nu * nv * 3);
    let mut uvs = if opts.uvs {
        Vec::with_capacity(nu * nv * 2)
    } else {
        Vec::new()
    };

    for &u in params_u {
        for &v in params_v {
            let p = surface.point(u, v);
            positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);

            let n = surface.normal(u, v).unwrap_or([0.0, 0.0, 1.0]);
            normals.extend_from_slice(&[n[0] as f32, n[1] as f32, n[2] as f32]);

            if opts.uvs {
                uvs.extend_from_slice(&[((u - u0) / span_u) as f32, ((v - v0) / span_v) as f32]);
            }
        }
    }

    let idx = |i: usize, j: usize| (i * nv + j) as u32;
    let pos_at = |i: usize, j: usize| {
        let o = (i * nv + j) * 3;
        [
            positions[o] as f64,
            positions[o + 1] as f64,
            positions[o + 2] as f64,
        ]
    };

    let mut indices = Vec::with_capacity((nu - 1) * (nv - 1) * 6);
    for i in 0..nu.saturating_sub(1) {
        for j in 0..nv.saturating_sub(1) {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            let (pa, pb, pc, pd) = (
                pos_at(i, j),
                pos_at(i + 1, j),
                pos_at(i + 1, j + 1),
                pos_at(i, j + 1),
            );
            if !degenerate(pa, pb, pc) {
                indices.extend_from_slice(&[a, b, c]);
            }
            if !degenerate(pa, pc, pd) {
                indices.extend_from_slice(&[a, c, d]);
            }
        }
    }

    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(positions, 3));
    g.set_attribute("normal", BufferAttribute::new(normals, 3));
    if opts.uvs {
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
    }
    g.set_index(indices);
    g
}

/// Zero-area to within a *relative* threshold — an absolute one would call
/// every triangle of a millimetre-scale model degenerate.
fn degenerate(a: V3, b: V3, c: V3) -> bool {
    let ab = v3::sub(b, a);
    let ac = v3::sub(c, a);
    let area2 = v3::norm(v3::cross(ab, ac));
    let scale = v3::norm(ab).max(v3::norm(ac)).max(v3::dist(b, c));
    area2 <= 1e-12 * scale * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::construct;

    fn max_radial_error(g: &BufferGeometry, center: V3, radius: f64) -> f64 {
        let pos = g.get_attribute("position").unwrap();
        pos.array
            .chunks_exact(3)
            .map(|c| {
                let p = [c[0] as f64, c[1] as f64, c[2] as f64];
                (v3::dist(p, center) - radius).abs()
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn flat_patch_needs_almost_no_samples() {
        let s = construct::plane([0.0; 3], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]);
        let (pu, pv) = sample_grid(&s, &TessellationOptions::with_tolerance(1e-4));
        assert_eq!(pu.len(), 2, "a plane is exactly a chord in u");
        assert_eq!(pv.len(), 2, "a plane is exactly a chord in v");
    }

    #[test]
    fn tighter_tolerance_produces_more_samples() {
        let s = construct::sphere([0.0; 3], 1.0);
        let coarse = sample_grid(&s, &TessellationOptions::with_tolerance(1e-2));
        let fine = sample_grid(&s, &TessellationOptions::with_tolerance(1e-5));
        assert!(
            fine.0.len() > coarse.0.len(),
            "u: {} vs {}",
            fine.0.len(),
            coarse.0.len()
        );
        assert!(
            fine.1.len() > coarse.1.len(),
            "v: {} vs {}",
            fine.1.len(),
            coarse.1.len()
        );
    }

    #[test]
    fn sphere_vertices_lie_on_the_sphere() {
        let s = construct::sphere([0.0; 3], 2.0);
        let g = tessellate_surface(&s, &TessellationOptions::with_tolerance(1e-3));
        // Positions are f32 in the buffer, so the floor is f32 epsilon times the
        // radius (~2e-7), not f64 exactness.
        assert!(max_radial_error(&g, [0.0; 3], 2.0) < 1e-5);
    }

    #[test]
    fn sphere_grid_edges_honour_the_tolerance() {
        // The guarantee is on the *grid* edges, the ones the sampler controls.
        // A quad's diagonal spans both directions at once and can deviate more;
        // that is a property of the quad split, not of the sampling, so measure
        // what is actually claimed.
        for &tol in &[1e-2, 1e-3] {
            let s = construct::sphere([0.0; 3], 2.0);
            let opts = TessellationOptions::with_tolerance(tol);
            let (pu, pv) = sample_grid(&s, &opts);
            assert!(
                sample_grid_meets_tolerance(&s, &pu, &pv, tol),
                "tolerance {tol} not met — max_samples bound first"
            );

            let mut worst = 0.0f64;
            for w in pu.windows(2) {
                for &v in &pv {
                    let mid = v3::scale(v3::add(s.point(w[0], v), s.point(w[1], v)), 0.5);
                    worst = worst.max((2.0 - v3::norm(mid)).abs());
                }
            }
            for w in pv.windows(2) {
                for &u in &pu {
                    let mid = v3::scale(v3::add(s.point(u, w[0]), s.point(u, w[1])), 0.5);
                    worst = worst.max((2.0 - v3::norm(mid)).abs());
                }
            }
            assert!(
                worst <= tol,
                "chord deviation {worst} exceeds tolerance {tol}"
            );
        }
    }

    #[test]
    fn a_bound_max_samples_is_reported_not_hidden() {
        let s = construct::sphere([0.0; 3], 100.0);
        let opts = TessellationOptions {
            tolerance: 1e-6,
            max_samples: 16,
            ..TessellationOptions::default()
        };
        let (pu, pv) = sample_grid(&s, &opts);
        assert!(pu.len() <= 16 && pv.len() <= 16);
        assert!(
            !sample_grid_meets_tolerance(&s, &pu, &pv, opts.tolerance),
            "a truncated grid must not claim to meet the tolerance"
        );
    }

    #[test]
    fn tessellated_normals_are_unit_and_radial_on_a_sphere() {
        let s = construct::sphere([0.0; 3], 3.0);
        let g = tessellate_surface(&s, &TessellationOptions::with_tolerance(1e-3));
        let pos = &g.get_attribute("position").unwrap().array;
        let nor = &g.get_attribute("normal").unwrap().array;
        assert_eq!(pos.len(), nor.len());

        for (p, n) in pos.chunks_exact(3).zip(nor.chunks_exact(3)) {
            let n = [n[0] as f64, n[1] as f64, n[2] as f64];
            assert!((v3::norm(n) - 1.0).abs() < 1e-6, "normal not unit: {n:?}");
            let radial = v3::normalize([p[0] as f64, p[1] as f64, p[2] as f64]).unwrap();
            assert!(v3::dot(n, radial) > 1.0 - 1e-4, "normal not radial");
        }
    }

    #[test]
    fn pole_triangles_are_dropped_not_emitted_degenerate() {
        let s = construct::sphere([0.0; 3], 1.0);
        let g = tessellate_surface(&s, &TessellationOptions::with_tolerance(5e-3));
        let pos = &g.get_attribute("position").unwrap().array;
        let idx = g.index.as_ref().unwrap();

        for tri in idx.chunks_exact(3) {
            let p: Vec<V3> = tri
                .iter()
                .map(|&i| {
                    let o = i as usize * 3;
                    [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
                })
                .collect();
            assert!(
                !degenerate(p[0], p[1], p[2]),
                "degenerate triangle survived"
            );
        }
        assert!(!idx.is_empty());
    }

    #[test]
    fn uvs_span_the_unit_square() {
        let s = construct::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0);
        let g = tessellate_surface(&s, &TessellationOptions::with_tolerance(1e-2));
        let uv = &g.get_attribute("uv").unwrap().array;
        let (mut lo_u, mut hi_u, mut lo_v, mut hi_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for c in uv.chunks_exact(2) {
            lo_u = lo_u.min(c[0]);
            hi_u = hi_u.max(c[0]);
            lo_v = lo_v.min(c[1]);
            hi_v = hi_v.max(c[1]);
        }
        assert!(
            lo_u.abs() < 1e-6 && (hi_u - 1.0).abs() < 1e-6,
            "u ∈ [{lo_u}, {hi_u}]"
        );
        assert!(
            lo_v.abs() < 1e-6 && (hi_v - 1.0).abs() < 1e-6,
            "v ∈ [{lo_v}, {hi_v}]"
        );
    }

    #[test]
    fn uvs_can_be_suppressed() {
        let s = construct::sphere([0.0; 3], 1.0);
        let opts = TessellationOptions {
            uvs: false,
            ..TessellationOptions::with_tolerance(1e-2)
        };
        let g = tessellate_surface(&s, &opts);
        assert!(g.get_attribute("uv").is_none());
    }

    #[test]
    fn max_samples_is_respected() {
        let s = construct::sphere([0.0; 3], 1000.0);
        // Breakpoints are mandatory, so the effective floor is their count.
        let floor_u = s.breakpoints_u().len();
        let floor_v = s.breakpoints_v().len();
        for &cap in &[4usize, 24, 100, 1024] {
            let opts = TessellationOptions {
                tolerance: 1e-9,
                max_samples: cap,
                ..TessellationOptions::default()
            };
            let (pu, pv) = sample_grid(&s, &opts);
            assert!(pu.len() <= cap.max(floor_u), "cap {cap}, u: {}", pu.len());
            assert!(pv.len() <= cap.max(floor_v), "cap {cap}, v: {}", pv.len());
        }
    }

    #[test]
    fn min_samples_is_respected() {
        let s = construct::plane([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let opts = TessellationOptions {
            min_samples: 9,
            ..TessellationOptions::default()
        };
        let (pu, pv) = sample_grid(&s, &opts);
        assert!(pu.len() >= 9, "u: {}", pu.len());
        assert!(pv.len() >= 9, "v: {}", pv.len());
    }

    #[test]
    fn curve_tessellation_honours_its_tolerance() {
        let c = construct::circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 5.0);
        for &tol in &[1e-2, 1e-3, 1e-4] {
            let pts = tessellate_curve(&c, &TessellationOptions::with_tolerance(tol));
            let mut worst = 0.0f64;
            for w in pts.windows(2) {
                let mid = v3::scale(v3::add(w[0], w[1]), 0.5);
                worst = worst.max((5.0 - v3::norm(mid)).abs());
            }
            assert!(
                worst <= tol * 1.5,
                "sagitta {worst} exceeds tolerance {tol}"
            );
        }
    }

    #[test]
    fn geometry_is_indexed_and_consistent() {
        let s = construct::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        let g = tessellate_surface(&s, &TessellationOptions::with_tolerance(1e-3));
        let n_verts = g.get_attribute("position").unwrap().count();
        let idx = g.index.as_ref().unwrap();
        assert!(idx.iter().all(|&i| (i as usize) < n_verts));
        assert_eq!(idx.len() % 3, 0);
        assert_eq!(g.get_attribute("normal").unwrap().count(), n_verts);
        assert_eq!(g.get_attribute("uv").unwrap().count(), n_verts);
    }
}
