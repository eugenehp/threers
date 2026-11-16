//! The boundary element method: fields in a region known from its surface
//! alone.
//!
//! Both solvers are collocation with piecewise-constant elements — one
//! unknown per triangle, at its centroid. That is the simplest BEM there is,
//! and converges like `h`: halving the triangle size halves the error. Use
//! a surface fine enough to resolve what you are after, and for waves, at
//! least six triangles per wavelength (ten for accuracy).
//!
//! - [`Laplace`]: `∇²u = 0` inside or outside a closed surface, with the
//!   potential or its normal derivative given on each triangle. Capacitance,
//!   electrostatic fields, potential flow, and conduction in a homogeneous
//!   body with no sources.
//! - [`Helmholtz`]: `∇²p + k²p = 0` outside a closed surface, radiating to
//!   infinity: the sound a vibrating surface makes, or how a rigid body
//!   scatters an incoming wave.
//!
//! # Conventions
//!
//! The surface's normals point *out of the body* — counter-clockwise
//! winding seen from outside, as three.js geometry has it — whether the
//! problem is inside or outside. "Flux" is `∂u/∂n` along that normal. A
//! surface wound inside out is flipped on the way in.
//!
//! # Cost
//!
//! The matrices are dense: `N²` entries and an `N³/3` LU for `N` triangles.
//! Two thousand triangles is about a second; ten thousand is minutes and a
//! gigabyte, which is where a fast multipole method would take over. With
//! `parallel`, assembly and field evaluation run across threads.

pub mod helmholtz;
pub mod laplace;

pub use helmholtz::{Helmholtz, HelmholtzSolution};
pub use laplace::{Laplace, LaplaceSolution, Potential};

use crate::surface::TriSurface;
use crate::vec3::*;
use crate::FemError;

/// A surface prepared for constant-element BEM.
#[derive(Clone, Debug)]
pub(crate) struct Elements {
    pub tris: Vec<[V3; 3]>,
    pub centroids: Vec<V3>,
    pub normals: Vec<V3>,
    pub areas: Vec<f64>,
    /// Longest edge per triangle.
    pub sizes: Vec<f64>,
    pub surface: TriSurface,
}

impl Elements {
    pub fn new(surface: &TriSurface) -> Result<Self, FemError> {
        if surface.triangles.is_empty() {
            return Err(FemError::InvalidSurface("no triangles"));
        }
        let mut surface = surface.clone();
        if surface.volume() < 0.0 {
            for t in &mut surface.triangles {
                t.swap(1, 2);
            }
        }
        let n = surface.triangles.len();
        let mut e = Self {
            tris: Vec::with_capacity(n),
            centroids: Vec::with_capacity(n),
            normals: Vec::with_capacity(n),
            areas: Vec::with_capacity(n),
            sizes: Vec::with_capacity(n),
            surface: TriSurface::default(),
        };
        for t in 0..n {
            let tri = surface.triangle(t);
            let (nrm, area) = surface.normal_area(t);
            if area <= 0.0 {
                return Err(FemError::InvalidSurface("a triangle has zero area"));
            }
            e.centroids.push(scale(add(add(tri[0], tri[1]), tri[2]), 1.0 / 3.0));
            e.normals.push(nrm);
            e.areas.push(area);
            e.sizes.push(
                dist(tri[0], tri[1])
                    .max(dist(tri[1], tri[2]))
                    .max(dist(tri[2], tri[0])),
            );
            e.tris.push(tri);
        }
        e.surface = surface;
        Ok(e)
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }
}

/// Dunavant's seven-point rule, exact to degree 5: barycentric points and
/// weights summing to 1.
const RULE7: [([f64; 3], f64); 7] = {
    const A1: f64 = 0.059_715_871_789_770;
    const B1: f64 = 0.470_142_064_105_115;
    const W1: f64 = 0.132_394_152_788_506;
    const A2: f64 = 0.797_426_985_353_087;
    const B2: f64 = 0.101_286_507_323_456;
    const W2: f64 = 0.125_939_180_544_827;
    [
        ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
        ([A1, B1, B1], W1),
        ([B1, A1, B1], W1),
        ([B1, B1, A1], W1),
        ([A2, B2, B2], W2),
        ([B2, A2, B2], W2),
        ([B2, B2, A2], W2),
    ]
};

/// Three-point rule, exact to degree 2.
const RULE3: [([f64; 3], f64); 3] = [
    ([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], 1.0 / 3.0),
    ([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], 1.0 / 3.0),
    ([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], 1.0 / 3.0),
];

/// `∫_tri f(y) dS_y` for a kernel that is smooth on the triangle but may
/// vary quickly when `x` is near it: the triangle is split into four,
/// recursively, until each piece is far from `x` against its own size.
pub(crate) fn integrate<T, F>(x: V3, tri: [V3; 3], f: &F) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default,
    F: Fn(V3) -> T,
{
    integrate_depth(x, tri, f, 0)
}

fn integrate_depth<T, F>(x: V3, tri: [V3; 3], f: &F, depth: u32) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default,
    F: Fn(V3) -> T,
{
    let [a, b, c] = tri;
    let size = dist(a, b).max(dist(b, c)).max(dist(c, a));
    let centroid = scale(add(add(a, b), c), 1.0 / 3.0);
    let d = dist(x, centroid);
    if d < 1.5 * size && depth < 5 {
        let ab = lerp(a, b, 0.5);
        let bc = lerp(b, c, 0.5);
        let ca = lerp(c, a, 0.5);
        let mut s = integrate_depth(x, [a, ab, ca], f, depth + 1);
        s = s + integrate_depth(x, [ab, b, bc], f, depth + 1);
        s = s + integrate_depth(x, [ca, bc, c], f, depth + 1);
        s = s + integrate_depth(x, [ab, bc, ca], f, depth + 1);
        return s;
    }
    let area = 0.5 * norm(cross(sub(b, a), sub(c, a)));
    let rule: &[([f64; 3], f64)] = if d < 8.0 * size { &RULE7 } else { &RULE3 };
    let mut s = T::default();
    for (l, w) in rule {
        let y = add(add(scale(a, l[0]), scale(b, l[1])), scale(c, l[2]));
        s = s + f(y) * (w * area);
    }
    s
}

/// `∫_tri f(y) dS_y` with the seven-point rule and no refinement — for
/// kernels with no singularity anywhere, such as far-field phases.
pub(crate) fn integrate_far<T, F>(tri: [V3; 3], f: &F) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default,
    F: Fn(V3) -> T,
{
    let [a, b, c] = tri;
    let area = 0.5 * norm(cross(sub(b, a), sub(c, a)));
    let mut s = T::default();
    for (l, w) in &RULE7 {
        let y = add(add(scale(a, l[0]), scale(b, l[1])), scale(c, l[2]));
        s = s + f(y) * (w * area);
    }
    s
}

/// `∫_tri dS / |x − y|` for `x` in the triangle's plane and inside it, in
/// closed form: the triangle seen from `x` is three triangles with apex at
/// `x`, and over each `∫ dS/r = h·[asinh(s/h)]` along its far edge.
pub(crate) fn inv_r_self(x: V3, tri: [V3; 3]) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let a = tri[k];
        let b = tri[(k + 1) % 3];
        let t = normalize(sub(b, a));
        let foot = add(a, scale(t, dot(sub(x, a), t)));
        let h = dist(x, foot);
        if h <= 1e-300 {
            continue;
        }
        let sa = dot(sub(a, foot), t);
        let sb = dot(sub(b, foot), t);
        s += h * ((sb / h).asinh() - (sa / h).asinh());
    }
    s
}

/// The singular self-term `∫ (smooth kernel) dS` over a triangle with the
/// collocation point at its centroid: three sub-triangles meeting at the
/// centroid, each integrated with subdivision, so the kink at `r = 0` falls
/// on their shared corner.
pub(crate) fn integrate_self<T, F>(tri: [V3; 3], f: &F) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default,
    F: Fn(V3) -> T,
{
    let c = scale(add(add(tri[0], tri[1]), tri[2]), 1.0 / 3.0);
    let mut s = T::default();
    for k in 0..3 {
        s = s + integrate_corner([c, tri[k], tri[(k + 1) % 3]], f, 0);
    }
    s
}

/// Integrate with refinement toward corner 0.
fn integrate_corner<T, F>(tri: [V3; 3], f: &F, depth: u32) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> + Default,
    F: Fn(V3) -> T,
{
    let [a, b, c] = tri;
    let area = 0.5 * norm(cross(sub(b, a), sub(c, a)));
    let rule = |t: [V3; 3], area: f64| {
        let mut s = T::default();
        for (l, w) in &RULE7 {
            let y = add(add(scale(t[0], l[0]), scale(t[1], l[1])), scale(t[2], l[2]));
            s = s + f(y) * (w * area);
        }
        s
    };
    if depth >= 6 {
        return rule(tri, area);
    }
    // Split off the corner: the small triangle at `a` recurses; the
    // trapezoid beyond it is two triangles away from the kink.
    let ab = lerp(a, b, 0.5);
    let ac = lerp(a, c, 0.5);
    let near = integrate_corner([a, ab, ac], f, depth + 1);
    let q1 = [ab, b, c];
    let q2 = [ab, c, ac];
    let a1 = 0.5 * norm(cross(sub(q1[1], q1[0]), sub(q1[2], q1[0])));
    let a2 = 0.5 * norm(cross(sub(q2[1], q2[0]), sub(q2[2], q2[0])));
    near + rule(q1, a1) + rule(q2, a2)
}

/// Run `f` per row, across threads with `parallel`.
pub(crate) fn rows<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    #[allow(unreachable_code)]
    (0..n).map(f).collect()
}

/// Points well inside a closed surface, for CHIEF: quasi-random over the
/// bounding box, kept when the winding number says inside and the nearest
/// centroid is at least a few element sizes away.
pub(crate) fn interior_points(e: &Elements, count: usize) -> Vec<V3> {
    let (lo, hi) = e.surface.bounds();
    let ext = sub(hi, lo);
    let mean_size = e.sizes.iter().sum::<f64>() / e.len() as f64;
    let mut out = Vec::new();
    // R3 low-discrepancy sequence (Roberts).
    let g = 1.220_744_084_605_759_5_f64;
    let alpha = [1.0 / g, 1.0 / (g * g), 1.0 / (g * g * g)];
    let mut clearance = 2.0 * mean_size;
    let mut i = 0u64;
    while out.len() < count && i < 20_000 {
        i += 1;
        let u = [0, 1, 2].map(|k| (0.5 + alpha[k] * i as f64).fract());
        let p = [0, 1, 2].map(|k| lo[k] + ext[k] * (0.05 + 0.9 * u[k]));
        if e.surface.winding_number(p) < 0.5 {
            continue;
        }
        let d = e.centroids.iter().map(|c| dist(*c, p)).fold(f64::INFINITY, f64::min);
        if d >= clearance {
            out.push(p);
        }
        if i.is_multiple_of(5000) {
            // A thin body has no point that far in; settle for less.
            clearance *= 0.5;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_self_integral_matches_refined_quadrature() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.3, 1.5, 0.0]];
        let c = scale(add(add(tri[0], tri[1]), tri[2]), 1.0 / 3.0);
        let exact = inv_r_self(c, tri);
        let numeric: f64 = integrate_self(tri, &|y: V3| 1.0 / dist(c, y));
        assert!((exact - numeric).abs() < 1e-3 * exact, "{exact} vs {numeric}");
    }

    #[test]
    fn subdivision_integrates_near_kernels() {
        // Solid angle of a unit square at height 0.05 above its centre,
        // integrated as two triangles: ∫ z / r³ dS.
        let z = 0.05;
        let x = [0.5, 0.5, z];
        let f = |y: V3| z / dist(x, y).powi(3);
        let s: f64 = integrate(x, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], &f)
            + integrate(x, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], &f);
        // Solid angle of an a×b rectangle seen from height d over its
        // centre: Ω = 4 asin(sin α sin β), α = atan(a/2d), β = atan(b/2d).
        let al = (0.5 / z).atan();
        let want = 4.0 * (al.sin() * al.sin()).asin();
        assert!((s - want).abs() < 2e-3 * want, "{s} vs {want}");
    }
}
