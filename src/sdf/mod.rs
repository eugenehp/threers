//! Signed distance fields: primitives, the operators that join them, and the
//! same library in WGSL so a shader and the CPU agree on every shape.
//!
//! A signed distance field answers one question about a point — how far is it
//! from the surface, negative inside — and that one number is enough to draw a
//! shape anti-aliased at any size (coverage is `0.5 - d` in pixels), to march a
//! ray to it, to mesh it, and to melt shapes into each other with a smooth
//! minimum. That last one is what makes them worth having: two capsules joined
//! with [`smooth_union`] meet in a fillet with no seam, which no polygon
//! pipeline gives you without a remesh.
//!
//! # Layout
//!
//! | | |
//! |---|---|
//! | [`d2`] | circle, box, rounded box, segment, capsule, tapered capsule, polygon |
//! | [`d3`] | sphere, ellipsoid, box, rounded box, capsule, tapered capsule, torus, cylinder, plane |
//! | this module | union / intersection / subtraction, their smooth forms, [`smooth_union_blend`] for carrying attributes through a blend, [`round`], [`onion`] |
//! | coverage | [`coverage`], [`coverage_edge`] with a pixel's [`footprint`] for edges at an angle, [`coverage_stroke`] for lines thinner than a pixel |
//! | [`Sdf2`], [`Sdf3`] | any `Fn(p) -> f32` is a field; gradients, normals, [`Sdf3::raymarch`], [`Sdf3::to_geometry`] (marching cubes) and [`Sdf3::to_surface_nets`] come with it |
//! | [`Shape`] | a field as a tree built at run time — and from JavaScript, as `THREE.SDF` — with conservative bounds |
//! | [`WGSL`] | the primitives and operators above, as `sdf2_*`, `sdf3_*` and `sdf_*` functions to splice into a shader |
//!
//! Symmetric primitives are centred on the origin, the way every SDF reference
//! writes them — translate by subtracting from the point:
//! `d2::circle(p - centre, r)`. Primitives defined by endpoints (segments,
//! capsules) take the endpoints, since a capsule has no natural centre.
//!
//! ```
//! use threers::math::Vector3;
//! use threers::sdf::{self, d3, Sdf3};
//!
//! // A ball melted into a bar.
//! let field = |p: Vector3| {
//!     let ball = d3::sphere(p - Vector3::new(0.0, 0.6, 0.0), 0.5);
//!     let bar = d3::capsule(p, Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), 0.2);
//!     sdf::smooth_union(ball, bar, 0.3)
//! };
//! assert!(field.distance(Vector3::new(0.0, 0.3, 0.0)) < 0.0); // in the fillet
//!
//! let hit = field
//!     .raymarch(Vector3::new(0.0, 0.6, 5.0), Vector3::new(0.0, 0.0, -1.0), Default::default())
//!     .expect("the ray points at the ball");
//! assert!((hit.distance - 4.5).abs() < 1e-3);
//! assert!(hit.normal.z > 0.99);
//!
//! let mesh = field.to_geometry(48, Vector3::ZERO, Vector3::new(2.6, 1.8, 1.2));
//! assert!(mesh.triangle_count() > 0);
//! ```
//!
//! # Exact and bounded
//!
//! Every primitive here is *exact*: the value is the true Euclidean distance,
//! outside and in. Union of exact fields is exact outside; intersection,
//! subtraction and the smooth operators are only *bounds* — they never
//! overestimate the distance, so sphere tracing stays safe, but they may
//! underestimate, so a ray takes more steps near them and an offset
//! (`round`, `onion`) of one is not a true offset. That is the usual trade, and
//! the reason [`Sdf3::raymarch`] counts its steps.
//!
//! # On the GPU
//!
//! [`WGSL`] is plain WGSL with no bindings and no entry points, so it can be
//! prepended to a [`ShaderMaterial`](crate::ShaderMaterial) fragment, a
//! compute shader, or anything else naga will take:
//!
//! ```
//! let fragment = format!("{}\n{}", threers::sdf::WGSL, r#"
//! @fragment
//! fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
//!     let d = sdf2_circle(in.uv - vec2<f32>(0.5), 0.3);
//!     let a = clamp(0.5 - d / fwidth(d), 0.0, 1.0);
//!     return vec4<f32>(vec3<f32>(a), 1.0);
//! }
//! "#);
//! let _material = threers::ShaderMaterial::new(fragment);
//! ```
//!
//! The two sides are tested against each other on a real adapter
//! (`tests/sdf_gpu.rs`), so a fix to one that misses the other fails there.

pub mod d2;
pub mod d3;
mod shape;
#[cfg(all(target_arch = "wasm32", not(threers_no_js_bindings)))]
mod web;

pub use shape::Shape;
#[cfg(all(target_arch = "wasm32", not(threers_no_js_bindings)))]
pub use web::WebSdf;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::extras::MarchingCubes;
use crate::math::{Vector2, Vector3};

/// The primitives and operators as WGSL: `sdf2_*` in 2D, `sdf3_*` in 3D and
/// `sdf_*` for the operators, which work in either. No bindings, no entry
/// points — prepend it to your own source. See the [module docs](self).
pub const WGSL: &str = include_str!("sdf.wgsl");

/// Union: the nearer of two surfaces. Exact outside both.
#[inline]
pub fn union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Intersection: inside both. A bound, not exact.
#[inline]
pub fn intersection(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// `a` with `b` cut out of it. A bound, not exact.
#[inline]
pub fn subtraction(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// Union with the seam filleted over a width of about `k`.
///
/// The polynomial smooth minimum: within `k` of each other the two fields
/// blend, and further apart it is [`union`]. `k <= 0` is a hard union. The
/// result is never above the hard union, so it stays a safe bound.
#[inline]
pub fn smooth_union(a: f32, b: f32, k: f32) -> f32 {
    smooth_union_blend(a, b, k).0
}

/// [`smooth_union`], and how much of the result is `a` (`1`) versus `b` (`0`).
///
/// The second value is what lets anything carried with a field — a colour, a
/// material id, a thickness for shading — blend through the same fillet the
/// surface does: `attr = b_attr + (a_attr - b_attr) * h`.
#[inline]
pub fn smooth_union_blend(a: f32, b: f32, k: f32) -> (f32, f32) {
    if k <= 0.0 {
        return if a <= b { (a, 1.0) } else { (b, 0.0) };
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b + (a - b) * h - k * h * (1.0 - h), h)
}

/// Intersection with the crease rounded over about `k`.
#[inline]
pub fn smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    -smooth_union(-a, -b, k)
}

/// `a` with `b` cut out of it, the cut's edge rounded over about `k`.
#[inline]
pub fn smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    -smooth_union(-a, b, k)
}

/// The surface grown outward by `r` — rounds every convex corner by `r`.
#[inline]
pub fn round(d: f32, r: f32) -> f32 {
    d - r
}

/// A shell of thickness `2 * t` around the surface.
#[inline]
pub fn onion(d: f32, t: f32) -> f32 {
    d.abs() - t
}

/// Anti-aliased coverage of a field sampled `d` pixels from its edge: `1`
/// well inside, `0` well outside, linear across the pixel straddling it.
///
/// Right for an edge running along a pixel row or column. For one at an
/// angle use [`coverage_edge`] with the pixel's [`footprint`]; for anything
/// thinner than a pixel, [`coverage_stroke`].
#[inline]
pub fn coverage(d: f32) -> f32 {
    (0.5 - d).clamp(0.0, 1.0)
}

/// How wide a one-pixel square is measured across an edge whose field has
/// gradient `g`: `1` for an edge along a row or a column, `√2` at 45°. The
/// width the coverage ramp should span — a ramp one pixel wide on a diagonal
/// is too sharp, and stair-steps. `1` for a zero gradient.
#[inline]
pub fn footprint(g: Vector2) -> f32 {
    let l = g.length();
    if l > 1e-12 {
        (g.x.abs() + g.y.abs()) / l
    } else {
        1.0
    }
}

/// [`coverage`] with a ramp `footprint` pixels wide, from [`footprint`].
#[inline]
pub fn coverage_edge(d: f32, footprint: f32) -> f32 {
    (0.5 - d / footprint.max(1e-6)).clamp(0.0, 1.0)
}

/// Coverage of a pixel by a stroke `half_width` either side of a centre line,
/// sampled `u` pixels from that line: the share of the pixel's footprint
/// (`footprint` wide across the stroke, from [`footprint`]; `1` for one
/// along a row or column) that the stroke covers — a box filter.
///
/// For a stroke wider than the footprint this is [`coverage_edge`] on its
/// edge. Where they part is a stroke *thinner* than a pixel, which
/// `coverage(u - half_width)` draws both too dark and too wide — a
/// 0.4-pixel line comes out at 0.9 and spreads over two pixels. This peaks
/// at the line's width, and across the line the coverages always sum to the
/// width wherever it falls between pixels, so a thin line keeps one weight
/// as it moves instead of shimmering, and does not bead along a diagonal.
#[inline]
pub fn coverage_stroke(u: f32, half_width: f32, footprint: f32) -> f32 {
    let (u, h) = (u.abs(), half_width.max(0.0));
    let w = footprint.max(1e-6);
    let lo = (u - 0.5 * w).max(-h);
    let hi = (u + 0.5 * w).min(h);
    ((hi - lo) / w).clamp(0.0, 1.0)
}

/// A 2D signed distance field. Implemented for every `Fn(Vector2) -> f32`.
pub trait Sdf2 {
    fn distance(&self, p: Vector2) -> f32;

    /// The field's gradient by central differences over `h`. Unit length for
    /// an exact field away from its medial axis; shorter near a blend.
    fn gradient(&self, p: Vector2, h: f32) -> Vector2 {
        let dx = self.distance(Vector2::new(p.x + h, p.y)) - self.distance(Vector2::new(p.x - h, p.y));
        let dy = self.distance(Vector2::new(p.x, p.y + h)) - self.distance(Vector2::new(p.x, p.y - h));
        Vector2::new(dx, dy) * (0.5 / h)
    }
}

impl<F: Fn(Vector2) -> f32> Sdf2 for F {
    fn distance(&self, p: Vector2) -> f32 {
        self(p)
    }
}

/// Settings for [`Sdf3::raymarch`].
#[derive(Debug, Clone, Copy)]
pub struct Raymarch {
    /// Give up past this distance along the ray.
    pub max_distance: f32,
    /// Give up after this many steps — what a ray grazing a bound-only field
    /// would otherwise spend forever on.
    pub max_steps: u32,
    /// A hit is a point closer than this to the surface, scaled by the
    /// distance travelled so far, so far-off surfaces are not asked for more
    /// precision than f32 has at that range.
    pub epsilon: f32,
}

impl Default for Raymarch {
    fn default() -> Self {
        Self { max_distance: 100.0, max_steps: 256, epsilon: 1e-4 }
    }
}

/// Where a ray met a field.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Along the ray, in units of `direction` (normalised by the march).
    pub distance: f32,
    pub point: Vector3,
    /// The outward surface normal.
    pub normal: Vector3,
    pub steps: u32,
}

/// A 3D signed distance field. Implemented for every `Fn(Vector3) -> f32`.
pub trait Sdf3 {
    fn distance(&self, p: Vector3) -> f32;

    /// The outward unit normal at `p`, from four evaluations on a tetrahedron
    /// of size `h` — one fewer than central differences, and no axis bias.
    fn normal(&self, p: Vector3, h: f32) -> Vector3 {
        let k = [
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        let mut n = Vector3::ZERO;
        for e in k {
            n = n + e * self.distance(p + e * h);
        }
        n.normalize()
    }

    /// Sphere-trace the ray from `origin` along `direction` (need not be unit
    /// length). `None` if it leaves `max_distance` or runs out of steps
    /// without coming within `epsilon` of the surface. A ray starting inside
    /// hits at distance 0.
    fn raymarch(&self, origin: Vector3, direction: Vector3, settings: Raymarch) -> Option<RayHit> {
        let dir = direction.normalize();
        let mut t = 0.0f32;
        for step in 0..settings.max_steps {
            let p = origin + dir * t;
            let d = self.distance(p);
            let eps = settings.epsilon * t.max(1.0);
            if d < eps {
                let h = eps.max(1e-5);
                return Some(RayHit { distance: t, point: p, normal: self.normal(p, h), steps: step + 1 });
            }
            t += d;
            if t > settings.max_distance {
                break;
            }
        }
        None
    }

    /// The zero surface as an indexed, watertight triangle mesh: the field is
    /// sampled on a `resolution`³ grid over the box of `size` around `centre`
    /// and meshed by [`MarchingCubes`]. Normals are the field's gradient.
    /// Anything outside the box is cut off, so leave the surface a cell of
    /// margin.
    fn to_geometry(&self, resolution: usize, centre: Vector3, size: Vector3) -> BufferGeometry {
        // Marching cubes counts "inside" as above the iso value, which is a
        // signed distance with its sign flipped.
        let mut geometry = MarchingCubes::new(resolution, size)
            .extract(0.0, |x, y, z| -self.distance(Vector3::new(x, y, z) + centre));
        if centre != Vector3::ZERO {
            if let Some(position) = geometry.get_attribute("position") {
                let mut array = position.array.clone();
                for v in array.chunks_exact_mut(3) {
                    v[0] += centre.x;
                    v[1] += centre.y;
                    v[2] += centre.z;
                }
                geometry.set_attribute("position", BufferAttribute::new(array, 3));
            }
        }
        geometry
    }

    /// The zero surface as a smooth indexed mesh by surface nets: one vertex
    /// per grid cell the surface crosses, joined into a quad across every
    /// grid edge it crosses. `cell` is the grid spacing, over the box from
    /// `min` to `max`.
    ///
    /// Where [`to_geometry`](Self::to_geometry) puts vertices on grid edges,
    /// which leaves a mesh that shows its grid as faceting and slivers, here
    /// each vertex starts at the mean of its cell's edge crossings and is
    /// then pulled onto the surface along the gradient — so a coarse grid
    /// still gives an even, round skin, which is what a soft body or a
    /// refracting surface wants. Normals are the field's gradient at each
    /// vertex, and every triangle is wound counter-clockwise seen from
    /// outside. Like `to_geometry`, anything outside the box is cut off.
    fn to_surface_nets(&self, cell: f32, min: Vector3, max: Vector3) -> SurfaceNets {
        let g = cell.max(1e-6);
        let n = |lo: f32, hi: f32| (((hi - lo) / g).ceil() as usize + 1).max(2);
        let (nx, ny, nz) = (n(min.x, max.x), n(min.y, max.y), n(min.z, max.z));
        let id = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let at = |i: usize, j: usize, k: usize| min + Vector3::new(i as f32, j as f32, k as f32) * g;
        let mut field = vec![0.0f32; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field[id(i, j, k)] = self.distance(at(i, j, k));
                }
            }
        }
        const CORNER: [[usize; 3]; 8] =
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
        const EDGE: [[usize; 2]; 12] =
            [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
        let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
        let cid = |i: usize, j: usize, k: usize| (k * cy + j) * cx + i;
        let mut cell_vertex = vec![u32::MAX; cx * cy * cz];
        let mut position: Vec<Vector3> = Vec::new();
        for k in 0..cz {
            for j in 0..cy {
                for i in 0..cx {
                    let mut v = [0.0f32; 8];
                    let mut inside = 0;
                    for (c, o) in CORNER.iter().enumerate() {
                        v[c] = field[id(i + o[0], j + o[1], k + o[2])];
                        inside += (v[c] < 0.0) as u32;
                    }
                    if inside == 0 || inside == 8 {
                        continue;
                    }
                    let (mut sum, mut count) = (Vector3::ZERO, 0.0f32);
                    for [a, b] in EDGE {
                        if (v[a] < 0.0) == (v[b] < 0.0) {
                            continue;
                        }
                        let t = v[a] / (v[a] - v[b]);
                        let (pa, pb) = (CORNER[a], CORNER[b]);
                        let corner = |q: [usize; 3]| Vector3::new(q[0] as f32, q[1] as f32, q[2] as f32);
                        sum = sum + corner(pa) + (corner(pb) - corner(pa)) * t;
                        count += 1.0;
                    }
                    cell_vertex[cid(i, j, k)] = position.len() as u32;
                    position.push(at(i, j, k) + sum * (g / count));
                }
            }
        }
        // Onto the surface: a few Newton steps along the gradient.
        let h = g * 0.1;
        let mut normal = Vec::with_capacity(position.len());
        for p in position.iter_mut() {
            for _ in 0..3 {
                let d = self.distance(*p);
                let grad = self.normal(*p, h);
                if !grad.x.is_finite() {
                    break;
                }
                *p = *p - grad * d;
            }
            normal.push(self.normal(*p, h));
        }
        let mut index: Vec<u32> = Vec::new();
        let mut quad = |q: [u32; 4]| {
            if q.iter().all(|&v| v != u32::MAX) {
                index.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            }
        };
        let cv = |i: usize, j: usize, k: usize| cell_vertex[cid(i, j, k)];
        for k in 1..cz {
            for j in 1..cy {
                for i in 0..cx {
                    if (field[id(i, j, k)] < 0.0) != (field[id(i + 1, j, k)] < 0.0) {
                        quad([cv(i, j - 1, k - 1), cv(i, j, k - 1), cv(i, j, k), cv(i, j - 1, k)]);
                    }
                }
            }
        }
        for k in 1..cz {
            for j in 0..cy {
                for i in 1..cx {
                    if (field[id(i, j, k)] < 0.0) != (field[id(i, j + 1, k)] < 0.0) {
                        quad([cv(i - 1, j, k - 1), cv(i - 1, j, k), cv(i, j, k), cv(i, j, k - 1)]);
                    }
                }
            }
        }
        for k in 0..cz {
            for j in 1..cy {
                for i in 1..cx {
                    if (field[id(i, j, k)] < 0.0) != (field[id(i, j, k + 1)] < 0.0) {
                        quad([cv(i - 1, j - 1, k), cv(i, j - 1, k), cv(i, j, k), cv(i - 1, j, k)]);
                    }
                }
            }
        }
        // Counter-clockwise from outside: against the mean normal.
        for t in index.chunks_exact_mut(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = (position[b] - position[a]).cross(position[c] - position[a]);
            if face.dot(normal[a] + normal[b] + normal[c]) < 0.0 {
                t.swap(1, 2);
            }
        }
        SurfaceNets { position, normal, index }
    }
}

/// A mesh from [`Sdf3::to_surface_nets`], kept as plain arrays so a caller
/// can bind its vertices to something before it becomes a geometry.
#[derive(Debug, Clone, Default)]
pub struct SurfaceNets {
    pub position: Vec<Vector3>,
    pub normal: Vec<Vector3>,
    /// Triangles, three indices each.
    pub index: Vec<u32>,
}

impl SurfaceNets {
    /// As an indexed geometry with `position` and `normal`.
    pub fn to_geometry(&self) -> BufferGeometry {
        let flat = |v: &[Vector3]| v.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<f32>>();
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(flat(&self.position), 3));
        g.set_attribute("normal", BufferAttribute::new(flat(&self.normal), 3));
        g.set_index(self.index.clone());
        g
    }

    pub fn triangle_count(&self) -> usize {
        self.index.len() / 3
    }
}

impl<F: Fn(Vector3) -> f32> Sdf3 for F {
    fn distance(&self, p: Vector3) -> f32 {
        self(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_union_is_the_hard_union_beyond_k() {
        assert_eq!(smooth_union(1.0, 3.0, 0.5), 1.0);
        assert_eq!(smooth_union(3.0, 1.0, 0.5), 1.0);
        // At a tie it digs in by k/4, the depth of the fillet.
        assert!((smooth_union(1.0, 1.0, 0.8) - 0.8).abs() < 1e-6);
        // Never above the hard union, so ray marching stays safe.
        for i in 0..50 {
            let a = i as f32 * 0.05 - 1.0;
            for k in [0.1, 0.5, 2.0] {
                assert!(smooth_union(a, 0.2, k) <= a.min(0.2) + 1e-7);
            }
        }
    }

    #[test]
    fn a_zero_width_blend_is_a_hard_union() {
        assert_eq!(smooth_union_blend(0.3, 0.7, 0.0), (0.3, 1.0));
        assert_eq!(smooth_union_blend(0.7, 0.3, 0.0), (0.3, 0.0));
        assert_eq!(smooth_intersection(0.3, 0.7, 0.0), 0.7);
    }

    #[test]
    fn the_blend_weight_follows_the_nearer_field() {
        assert_eq!(smooth_union_blend(0.0, 5.0, 1.0).1, 1.0);
        assert_eq!(smooth_union_blend(5.0, 0.0, 1.0).1, 0.0);
        assert!((smooth_union_blend(2.0, 2.0, 1.0).1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn subtraction_and_intersection_signs() {
        // a: a unit disc; b: a disc of 0.5 at the same centre.
        let p = |r: f32| (r - 1.0, r - 0.5);
        let (a, b) = p(0.25);
        assert!(subtraction(a, b) > 0.0, "the hole is outside");
        let (a, b) = p(0.75);
        assert!(subtraction(a, b) < 0.0, "the ring is inside");
        assert!(intersection(a, b) > 0.0);
        assert!(smooth_subtraction(a, b, 0.05) < 0.0);
    }

    #[test]
    fn raymarch_finds_a_sphere_and_misses_beside_it() {
        let ball = |p: Vector3| d3::sphere(p, 1.0);
        let hit = ball
            .raymarch(Vector3::new(0.0, 0.0, -4.0), Vector3::new(0.0, 0.0, 2.0), Raymarch::default())
            .unwrap();
        assert!((hit.distance - 3.0).abs() < 1e-3, "{hit:?}");
        assert!((hit.normal.z + 1.0).abs() < 1e-3, "{hit:?}");
        assert!(ball
            .raymarch(Vector3::new(1.5, 0.0, -4.0), Vector3::new(0.0, 0.0, 1.0), Raymarch::default())
            .is_none());
        // Starting inside is a hit where it stands.
        let inside = ball.raymarch(Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), Raymarch::default()).unwrap();
        assert_eq!(inside.distance, 0.0);
    }

    #[test]
    fn meshing_an_offset_sphere_lands_where_the_field_is() {
        let c = Vector3::new(3.0, -1.0, 2.0);
        let ball = move |p: Vector3| d3::sphere(p - c, 0.8);
        let mut g = ball.to_geometry(32, c, Vector3::new(2.0, 2.0, 2.0));
        assert!(g.triangle_count() > 100);
        let b = g.compute_bounding_box();
        for (lo, hi, centre) in [(b.min.x, b.max.x, c.x), (b.min.y, b.max.y, c.y), (b.min.z, b.max.z, c.z)] {
            assert!((lo - (centre - 0.8)).abs() < 0.05, "{lo} vs {}", centre - 0.8);
            assert!((hi - (centre + 0.8)).abs() < 0.05, "{hi} vs {}", centre + 0.8);
        }
        // Every vertex on the surface, to within the grid's linear interpolation.
        for p in g.positions().unwrap() {
            assert!(ball(p).abs() < 0.01, "{p:?} is {} off", ball(p));
        }
    }

    /// Across a line, a box filter's coverages sum to the line's width
    /// wherever the line falls between pixel centres — what keeps a thin line
    /// one weight as it moves. Checked for thin and thick lines, along a row
    /// (footprint 1) and on a diagonal (footprint √2, pixels √½ apart across it).
    #[test]
    fn stroke_coverage_conserves_the_line_width() {
        for (w, step) in [(1.0f32, 1.0f32), (std::f32::consts::SQRT_2, std::f32::consts::FRAC_1_SQRT_2)] {
            for half in [0.05f32, 0.2, 0.35, 0.5, 1.3, 4.0] {
                for k in 0..10 {
                    let phase = k as f32 / 10.0 * step;
                    let total: f32 = (-20..=20).map(|i| coverage_stroke(i as f32 * step + phase, half, w)).sum();
                    // Exact, not approximate: the spacing divides the footprint, so
                    // the footprints tile the line's normal evenly and the sum
                    // integrates the stroke's width.
                    assert!((total * step - 2.0 * half).abs() < 1e-4 * (1.0 + half), "w {w} half {half} phase {phase}: {}", total * step);
                }
            }
        }
    }

    #[test]
    fn stroke_coverage_peaks_at_the_width_and_matches_edges_when_thick() {
        // a 0.4-pixel line centred on a pixel covers 0.4 of it, not 0.9
        assert!((coverage_stroke(0.0, 0.2, 1.0) - 0.4).abs() < 1e-6);
        assert!((coverage(0.0 - 0.2) - 0.7).abs() < 1e-6, "what the edge rule gives");
        assert_eq!(coverage_stroke(1.0, 0.2, 1.0), 0.0, "a pixel away it is gone");
        // thick: the same as the edge rule on the stroke's edge
        for i in 0..40 {
            let u = i as f32 * 0.1;
            assert!((coverage_stroke(u, 2.0, 1.0) - coverage(u - 2.0)).abs() < 1e-6, "{u}");
            assert!((coverage_stroke(u, 2.0, 1.3) - coverage_edge(u - 2.0, 1.3)).abs() < 1e-6, "{u}");
        }
    }

    #[test]
    fn the_footprint_widens_on_a_diagonal() {
        assert_eq!(footprint(Vector2::new(0.0, -3.0)), 1.0);
        assert!((footprint(Vector2::new(1.0, 1.0)) - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert_eq!(footprint(Vector2::ZERO), 1.0);
        assert_eq!(coverage_edge(0.0, 1.4), 0.5);
        assert!((coverage_edge(0.6, 1.4) - (0.5 - 0.6 / 1.4)).abs() < 1e-6, "still partly covered");
        assert_eq!(coverage(0.6), 0.0, "where the one-pixel ramp has already stopped");
    }

    #[test]
    fn gradients_are_unit_on_exact_fields() {
        let disc = |p: Vector2| d2::circle(p, 1.0);
        let g = disc.gradient(Vector2::new(2.0, 0.0), 1e-3);
        assert!((g.x - 1.0).abs() < 1e-3 && g.y.abs() < 1e-3);
        let n = (|p: Vector3| d3::cuboid(p, Vector3::new(1.0, 2.0, 3.0))).normal(Vector3::new(0.2, 2.5, 0.1), 1e-3);
        assert!((n.y - 1.0).abs() < 1e-3, "{n:?}");
    }
}
