//! Collision shapes and their mass properties.
//!
//! # Conventions
//!
//! - Capsules, cylinders and cones are **Y-axis aligned**, matching
//!   `CapsuleGeometry` / `CylinderGeometry` / `ConeGeometry` in three.js.
//! - Sizes are given as **half extents** (`cuboid`, `capsule`, …) because that
//!   is what the solver works in. Every such constructor has a
//!   `*_from_size` sibling taking the full dimensions a `*Geometry` would, so
//!   you never have to remember which one you are holding.
//! - Cones and cylinders are centred on the origin: a cone of half-height `h`
//!   has its base at `y = -h` and its apex at `y = +h`.

use crate::hull::ConvexHull;
use crate::math::{closest_point_on_segment, try_normalize, Aabb, Isometry, Mat3};
use crate::trimesh::TriMesh;
use std::sync::Arc;
use threers::core::BufferGeometry;
use threers::math::Vector3;

/// Stand-in for "infinite" in the AABB of an unbounded shape. Large enough that
/// nothing real escapes it, small enough that squaring it stays finite in `f32`.
const HUGE: f32 = 1.0e9;

/// The smallest dimension a shape may have. Zero-sized shapes produce NaN
/// normals and singular inertia tensors, so dimensions are clamped here rather
/// than panicking deep inside the solver.
const MIN_DIM: f32 = 1.0e-6;

/// How to turn a drawn mesh into something the solver can collide.
///
/// The mesh you render is rarely the shape you want to simulate: an exact
/// triangle soup is hollow and cannot back a dynamic body, while a box around a
/// gear is not a gear. This is the choice between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColliderFit {
    /// Convex hull of the mesh's vertices. The default for anything that moves:
    /// solid, cheap to test, and close to the drawn shape unless it is concave.
    #[default]
    ConvexHull,
    /// The triangles themselves. Exact and hollow — for fixed level geometry.
    TriMesh,
    /// Tight axis-aligned box.
    Box,
    /// Sphere enclosing the mesh.
    Ball,
    /// Y-aligned capsule around the mesh's footprint.
    Capsule,
    /// Y-aligned cylinder around the mesh's footprint.
    Cylinder,
}

/// A collision shape in its own local frame.
///
/// Cloning is cheap — the mesh-backed variants are behind an [`Arc`], so the
/// same hull or triangle mesh can be shared across thousands of bodies.
#[derive(Debug, Clone)]
pub enum Shape {
    /// Sphere centred on the origin.
    Ball { radius: f32 },
    /// Axis-aligned box centred on the origin.
    Cuboid { half_extents: Vector3 },
    /// Y-aligned capsule: a cylinder of half-length `half_height` capped with
    /// two hemispheres of `radius`. Total height is `2 * (half_height + radius)`.
    Capsule { half_height: f32, radius: f32 },
    /// Y-aligned cylinder centred on the origin.
    Cylinder { half_height: f32, radius: f32 },
    /// Y-aligned cone, base at `-half_height`, apex at `+half_height`.
    Cone { half_height: f32, radius: f32 },
    /// Infinite solid half-space. The solid side is the one the normal points
    /// *away* from, so `half_space(Vector3::UP)` is a floor at `y = 0`.
    HalfSpace { normal: Vector3 },
    /// Arbitrary convex polyhedron.
    ConvexHull(Arc<ConvexHull>),
    /// Triangle soup. Hollow — see [`crate::trimesh`] before using on a dynamic body.
    TriMesh(Arc<TriMesh>),
    /// Several shapes rigidly welded together, each with its own local offset.
    Compound(Arc<Vec<(Isometry, Shape)>>),
}

impl Shape {
    // ---- constructors -----------------------------------------------------

    pub fn ball(radius: f32) -> Self {
        Self::Ball {
            radius: radius.max(MIN_DIM),
        }
    }

    /// Box from **half** extents.
    pub fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self::Cuboid {
            half_extents: Vector3::new(hx.max(MIN_DIM), hy.max(MIN_DIM), hz.max(MIN_DIM)),
        }
    }

    /// Box from **full** width / height / depth, matching `BoxGeometry(w, h, d)`.
    pub fn cuboid_from_size(width: f32, height: f32, depth: f32) -> Self {
        Self::cuboid(width * 0.5, height * 0.5, depth * 0.5)
    }

    /// Capsule from the **half length of the cylinder section** and its radius.
    pub fn capsule(half_height: f32, radius: f32) -> Self {
        Self::Capsule {
            half_height: half_height.max(0.0),
            radius: radius.max(MIN_DIM),
        }
    }

    /// Capsule matching `CapsuleGeometry(radius, length)`, where `length` is the
    /// cylinder section excluding the two caps.
    pub fn capsule_from_size(radius: f32, length: f32) -> Self {
        Self::capsule(length * 0.5, radius)
    }

    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        Self::Cylinder {
            half_height: half_height.max(MIN_DIM),
            radius: radius.max(MIN_DIM),
        }
    }

    /// Cylinder matching `CylinderGeometry(r, r, height)`.
    pub fn cylinder_from_size(radius: f32, height: f32) -> Self {
        Self::cylinder(height * 0.5, radius)
    }

    pub fn cone(half_height: f32, radius: f32) -> Self {
        Self::Cone {
            half_height: half_height.max(MIN_DIM),
            radius: radius.max(MIN_DIM),
        }
    }

    /// Cone matching `ConeGeometry(radius, height)`.
    pub fn cone_from_size(radius: f32, height: f32) -> Self {
        Self::cone(height * 0.5, radius)
    }

    /// Infinite half-space. Solid on the side the normal points away from.
    pub fn half_space(normal: Vector3) -> Self {
        Self::HalfSpace {
            normal: try_normalize(normal).unwrap_or(Vector3::UP),
        }
    }

    /// A ground plane at `y = 0` — `half_space(Vector3::UP)`.
    pub fn ground() -> Self {
        Self::half_space(Vector3::UP)
    }

    /// Convex hull of a point cloud. `None` if the points are degenerate
    /// (fewer than four, collinear, or coplanar).
    pub fn convex_hull(points: &[Vector3]) -> Option<Self> {
        ConvexHull::from_points(points).map(|h| Self::ConvexHull(Arc::new(h)))
    }

    /// Convex hull of whatever you are drawing. The usual way to give a dynamic
    /// body a shape that follows its mesh.
    pub fn convex_hull_from_geometry(geometry: &BufferGeometry) -> Option<Self> {
        let pos = geometry.get_attribute("position")?;
        if pos.item_size < 3 {
            return None;
        }
        let pts: Vec<Vector3> = pos
            .array
            .chunks_exact(pos.item_size)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();
        Self::convex_hull(&pts)
    }

    pub fn trimesh(vertices: Vec<Vector3>, indices: Vec<[u32; 3]>) -> Option<Self> {
        TriMesh::new(vertices, indices).map(|m| Self::TriMesh(Arc::new(m)))
    }

    /// Exact triangle-mesh collider from a `threers` geometry. Best for static
    /// level geometry.
    pub fn trimesh_from_geometry(geometry: &BufferGeometry) -> Option<Self> {
        TriMesh::from_geometry(geometry).map(|m| Self::TriMesh(Arc::new(m)))
    }

    /// Weld several shapes into one rigid collider.
    pub fn compound(parts: Vec<(Isometry, Shape)>) -> Self {
        Self::Compound(Arc::new(parts))
    }

    /// The tight axis-aligned box of `geometry`, as a cuboid at the geometry's
    /// centre. The cheapest useful collider for a mesh.
    ///
    /// Returns the shape and the offset of the box centre from the geometry
    /// origin — pass that offset as the collider's local transform when the
    /// mesh is not centred.
    pub fn aabb_from_geometry(geometry: &BufferGeometry) -> Option<(Self, Vector3)> {
        let pos = geometry.get_attribute("position")?;
        if pos.item_size < 3 || pos.array.len() < 3 {
            return None;
        }
        let mut b = Aabb::empty();
        for c in pos.array.chunks_exact(pos.item_size) {
            b.expand_by_point(Vector3::new(c[0], c[1], c[2]));
        }
        let half = b.size() * 0.5;
        Some((Self::cuboid(half.x, half.y, half.z), b.center()))
    }

    /// Fit a collider shape to `geometry` the way `fit` asks for.
    ///
    /// Returns the shape and where to put it: a mesh drawn away from its own
    /// origin needs the primitive placed at the geometry's centre, and only the
    /// fitter knows where that is. `Isometry::IDENTITY` for the fits that keep
    /// the mesh's own vertices.
    pub fn fit_to_geometry(geometry: &BufferGeometry, fit: ColliderFit) -> Option<(Self, Isometry)> {
        match fit {
            ColliderFit::ConvexHull => {
                Some((Self::convex_hull_from_geometry(geometry)?, Isometry::IDENTITY))
            }
            ColliderFit::TriMesh => {
                Some((Self::trimesh_from_geometry(geometry)?, Isometry::IDENTITY))
            }
            ColliderFit::Box => {
                let (shape, center) = Self::aabb_from_geometry(geometry)?;
                Some((shape, Isometry::from_translation(center)))
            }
            ColliderFit::Ball | ColliderFit::Capsule | ColliderFit::Cylinder => {
                let (bounds, center) = Self::aabb_from_geometry(geometry)?;
                let Self::Cuboid { half_extents } = bounds else {
                    return None;
                };
                // Round the footprint rather than the whole box: a capsule or a
                // cylinder fitted to a mesh is being asked to stand up in it, so
                // the radius has to clear x and z and the height is y's alone.
                let radius = half_extents.x.max(half_extents.z);
                let shape = match fit {
                    ColliderFit::Ball => Self::ball(radius.max(half_extents.y)),
                    ColliderFit::Capsule => {
                        Self::capsule((half_extents.y - radius).max(0.0), radius)
                    }
                    _ => Self::cylinder(half_extents.y, radius),
                };
                Some((shape, Isometry::from_translation(center)))
            }
        }
    }

    // ---- queries ----------------------------------------------------------

    /// Whether the shape is convex, and so eligible for GJK/EPA and shape casts.
    pub fn is_convex(&self) -> bool {
        !matches!(
            self,
            Self::HalfSpace { .. } | Self::TriMesh(_) | Self::Compound(_)
        )
    }

    /// Whether the shape has infinite extent, and so is never culled by the
    /// broad phase.
    pub fn is_unbounded(&self) -> bool {
        match self {
            Self::HalfSpace { .. } => true,
            Self::Compound(parts) => parts.iter().any(|(_, s)| s.is_unbounded()),
            _ => false,
        }
    }

    /// Whether this shape can only ever back a fixed body. Half-spaces are
    /// infinite and triangle meshes are hollow; neither has usable inertia.
    pub fn is_static_only(&self) -> bool {
        match self {
            Self::HalfSpace { .. } => true,
            Self::TriMesh(m) => m.signed_volume() <= 0.0,
            Self::Compound(parts) => parts.iter().any(|(_, s)| s.is_static_only()),
            _ => false,
        }
    }

    /// AABB in the shape's own frame.
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Self::Ball { radius } => {
                let r = Vector3::new(*radius, *radius, *radius);
                Aabb::new(-r, r)
            }
            Self::Cuboid { half_extents } => Aabb::new(-*half_extents, *half_extents),
            Self::Capsule {
                half_height,
                radius,
            } => {
                let e = Vector3::new(*radius, half_height + radius, *radius);
                Aabb::new(-e, e)
            }
            Self::Cylinder {
                half_height,
                radius,
            }
            | Self::Cone {
                half_height,
                radius,
            } => {
                let e = Vector3::new(*radius, *half_height, *radius);
                Aabb::new(-e, e)
            }
            Self::HalfSpace { .. } => Aabb::new(
                Vector3::new(-HUGE, -HUGE, -HUGE),
                Vector3::new(HUGE, HUGE, HUGE),
            ),
            Self::ConvexHull(h) => h.aabb(),
            Self::TriMesh(m) => m.aabb(),
            Self::Compound(parts) => {
                let mut b = Aabb::empty();
                for (iso, shape) in parts.iter() {
                    b = b.union(&shape.compute_aabb(iso));
                }
                b
            }
        }
    }

    /// World-space AABB after `iso`.
    ///
    /// Exact for balls and boxes; conservative (never too small) for the
    /// rotated round shapes, which is all the broad phase requires.
    pub fn compute_aabb(&self, iso: &Isometry) -> Aabb {
        match self {
            Self::Ball { radius } => {
                let r = Vector3::new(*radius, *radius, *radius);
                Aabb::new(iso.translation - r, iso.translation + r)
            }
            Self::HalfSpace { .. } => Aabb::new(
                Vector3::new(-HUGE, -HUGE, -HUGE),
                Vector3::new(HUGE, HUGE, HUGE),
            ),
            Self::Capsule {
                half_height,
                radius,
            } => {
                // Exact: the capsule is a segment swept by a sphere.
                let axis = iso.transform_vector(Vector3::new(0.0, *half_height, 0.0));
                let r = Vector3::new(*radius, *radius, *radius);
                let (a, b) = (iso.translation - axis, iso.translation + axis);
                Aabb::new(a.min(b) - r, a.max(b) + r)
            }
            Self::Compound(parts) => {
                let mut b = Aabb::empty();
                for (local, shape) in parts.iter() {
                    b = b.union(&shape.compute_aabb(&iso.mul(local)));
                }
                b
            }
            _ => {
                let local = self.local_aabb();
                let mut b = Aabb::empty();
                for i in 0..8 {
                    let c = Vector3::new(
                        if i & 1 == 0 { local.min.x } else { local.max.x },
                        if i & 2 == 0 { local.min.y } else { local.max.y },
                        if i & 4 == 0 { local.min.z } else { local.max.z },
                    );
                    b.expand_by_point(iso.transform_point(c));
                }
                b
            }
        }
    }

    /// Support function: the farthest point of the shape along `dir`, in local
    /// space. `None` for non-convex shapes, which GJK cannot handle.
    ///
    /// `dir` need not be normalised.
    pub fn support_local(&self, dir: Vector3) -> Option<Vector3> {
        Some(match self {
            Self::Ball { radius } => try_normalize(dir).unwrap_or(Vector3::UP) * *radius,
            Self::Cuboid { half_extents } => Vector3::new(
                if dir.x >= 0.0 { half_extents.x } else { -half_extents.x },
                if dir.y >= 0.0 { half_extents.y } else { -half_extents.y },
                if dir.z >= 0.0 { half_extents.z } else { -half_extents.z },
            ),
            Self::Capsule {
                half_height,
                radius,
            } => {
                let y = if dir.y >= 0.0 { *half_height } else { -*half_height };
                Vector3::new(0.0, y, 0.0) + try_normalize(dir).unwrap_or(Vector3::UP) * *radius
            }
            Self::Cylinder {
                half_height,
                radius,
            } => {
                let y = if dir.y >= 0.0 { *half_height } else { -*half_height };
                let radial = try_normalize(Vector3::new(dir.x, 0.0, dir.z))
                    .map(|d| d * *radius)
                    .unwrap_or(Vector3::ZERO);
                Vector3::new(radial.x, y, radial.z)
            }
            Self::Cone {
                half_height,
                radius,
            } => {
                let apex = Vector3::new(0.0, *half_height, 0.0);
                let radial = try_normalize(Vector3::new(dir.x, 0.0, dir.z))
                    .map(|d| d * *radius)
                    .unwrap_or(Vector3::ZERO);
                let rim = Vector3::new(radial.x, -*half_height, radial.z);
                if apex.dot(dir) >= rim.dot(dir) {
                    apex
                } else {
                    rim
                }
            }
            Self::ConvexHull(h) => h.support(dir),
            Self::HalfSpace { .. } | Self::TriMesh(_) | Self::Compound(_) => return None,
        })
    }

    /// Is `p` (in local space) inside the solid?
    ///
    /// Triangle meshes are hollow, so this is always `false` for them.
    pub fn contains_point_local(&self, p: Vector3) -> bool {
        match self {
            Self::Ball { radius } => p.length_sq() <= radius * radius,
            Self::Cuboid { half_extents } => {
                p.x.abs() <= half_extents.x
                    && p.y.abs() <= half_extents.y
                    && p.z.abs() <= half_extents.z
            }
            Self::Capsule {
                half_height,
                radius,
            } => {
                let seg = closest_point_on_segment(
                    p,
                    Vector3::new(0.0, -*half_height, 0.0),
                    Vector3::new(0.0, *half_height, 0.0),
                );
                (p - seg).length_sq() <= radius * radius
            }
            Self::Cylinder {
                half_height,
                radius,
            } => p.y.abs() <= *half_height && p.x * p.x + p.z * p.z <= radius * radius,
            Self::Cone {
                half_height,
                radius,
            } => {
                if p.y.abs() > *half_height {
                    return false;
                }
                // Radius tapers linearly from `radius` at the base to 0 at the apex.
                let t = (half_height - p.y) / (2.0 * half_height);
                let r = radius * t;
                p.x * p.x + p.z * p.z <= r * r
            }
            Self::HalfSpace { normal } => normal.dot(p) <= 0.0,
            Self::ConvexHull(h) => h.contains_point(p),
            Self::TriMesh(_) => false,
            Self::Compound(parts) => parts
                .iter()
                .any(|(iso, s)| s.contains_point_local(iso.inverse_transform_point(p))),
        }
    }

    // ---- mass -------------------------------------------------------------

    /// Mass, centre of mass and inertia tensor at the given density
    /// (mass per unit volume; water is ~1000 in SI, but any consistent unit
    /// works).
    ///
    /// Unbounded and hollow shapes return [`MassProperties::ZERO`], which the
    /// body builder reads as "this must be a fixed body".
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        let density = density.max(0.0);
        match self {
            Self::Ball { radius } => {
                let r = *radius;
                let mass = density * (4.0 / 3.0) * std::f32::consts::PI * r * r * r;
                let i = 0.4 * mass * r * r;
                MassProperties::new(mass, Vector3::ZERO, Mat3::from_diagonal(Vector3::new(i, i, i)))
            }
            Self::Cuboid { half_extents } => {
                let h = *half_extents;
                let mass = density * 8.0 * h.x * h.y * h.z;
                let k = mass / 3.0;
                MassProperties::new(
                    mass,
                    Vector3::ZERO,
                    Mat3::from_diagonal(Vector3::new(
                        k * (h.y * h.y + h.z * h.z),
                        k * (h.x * h.x + h.z * h.z),
                        k * (h.x * h.x + h.y * h.y),
                    )),
                )
            }
            Self::Cylinder {
                half_height,
                radius,
            } => {
                let (hh, r) = (*half_height, *radius);
                let mass = density * std::f32::consts::PI * r * r * 2.0 * hh;
                let iy = 0.5 * mass * r * r;
                let ix = mass * (3.0 * r * r + 4.0 * hh * hh) / 12.0;
                MassProperties::new(
                    mass,
                    Vector3::ZERO,
                    Mat3::from_diagonal(Vector3::new(ix, iy, ix)),
                )
            }
            Self::Capsule {
                half_height,
                radius,
            } => {
                let (hh, r) = (*half_height, *radius);
                let pi = std::f32::consts::PI;
                // Cylinder section.
                let mc = density * pi * r * r * 2.0 * hh;
                let cyl_y = 0.5 * mc * r * r;
                let cyl_x = mc * (r * r / 4.0 + hh * hh / 3.0);
                // Two hemispherical caps, together a full sphere of mass `ms`
                // whose centres sit at y = ±hh.
                let ms = density * (4.0 / 3.0) * pi * r * r * r;
                let cap_y = 0.4 * ms * r * r;
                let cap_x = ms * (0.4 * r * r + hh * hh + 0.75 * hh * r);
                MassProperties::new(
                    mc + ms,
                    Vector3::ZERO,
                    Mat3::from_diagonal(Vector3::new(
                        cyl_x + cap_x,
                        cyl_y + cap_y,
                        cyl_x + cap_x,
                    )),
                )
            }
            Self::Cone {
                half_height,
                radius,
            } => {
                let (hh, r) = (*half_height, *radius);
                let h = 2.0 * hh;
                let mass = density * std::f32::consts::PI * r * r * h / 3.0;
                let iy = 0.3 * mass * r * r;
                // About the centre of mass, which sits a quarter of the way up
                // from the base — i.e. at y = -hh/2 in shape space.
                let ix = mass * (3.0 / 20.0 * r * r + 3.0 / 80.0 * h * h);
                MassProperties::new(
                    mass,
                    Vector3::new(0.0, -hh * 0.5, 0.0),
                    Mat3::from_diagonal(Vector3::new(ix, iy, ix)),
                )
            }
            Self::ConvexHull(h) => {
                let tris = h.faces.iter().map(|f| {
                    [
                        h.vertices[f.indices[0] as usize],
                        h.vertices[f.indices[1] as usize],
                        h.vertices[f.indices[2] as usize],
                    ]
                });
                mass_properties_from_triangles(tris, density)
            }
            Self::TriMesh(m) => {
                if m.signed_volume() <= 0.0 {
                    return MassProperties::ZERO;
                }
                mass_properties_from_triangles((0..m.triangle_count()).map(|i| m.triangle(i)), density)
            }
            Self::HalfSpace { .. } => MassProperties::ZERO,
            Self::Compound(parts) => {
                let mut total = MassProperties::ZERO;
                for (iso, shape) in parts.iter() {
                    total = total.merged(&shape.mass_properties(density).transformed_by(iso));
                }
                total
            }
        }
    }
}

/// Mass, centre of mass and inertia tensor of a body or shape.
///
/// `inertia` is expressed **about `center_of_mass`**, in the frame the
/// properties were computed in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassProperties {
    pub mass: f32,
    pub center_of_mass: Vector3,
    pub inertia: Mat3,
}

impl MassProperties {
    pub const ZERO: Self = Self {
        mass: 0.0,
        center_of_mass: Vector3::ZERO,
        inertia: Mat3::ZERO,
    };

    pub const fn new(mass: f32, center_of_mass: Vector3, inertia: Mat3) -> Self {
        Self {
            mass,
            center_of_mass,
            inertia,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mass <= 0.0
    }

    /// Rescale to an exact total mass, keeping the inertia distribution.
    pub fn with_mass(&self, mass: f32) -> Self {
        if self.mass <= 0.0 {
            return Self::new(mass, self.center_of_mass, Mat3::ZERO);
        }
        let k = mass / self.mass;
        Self::new(mass, self.center_of_mass, self.inertia.scaled(k))
    }

    /// Re-express in a parent frame.
    pub fn transformed_by(&self, iso: &Isometry) -> Self {
        Self::new(
            self.mass,
            iso.transform_point(self.center_of_mass),
            self.inertia.rotated(iso.rotation),
        )
    }

    /// Combine two sets expressed in the *same* frame.
    pub fn merged(&self, other: &Self) -> Self {
        let mass = self.mass + other.mass;
        if mass <= 0.0 {
            return Self::ZERO;
        }
        let com = (self.center_of_mass * self.mass + other.center_of_mass * other.mass)
            * (1.0 / mass);
        let inertia = self
            .inertia
            .add(&parallel_axis(self.mass, self.center_of_mass - com))
            .add(&other.inertia)
            .add(&parallel_axis(other.mass, other.center_of_mass - com));
        Self::new(mass, com, inertia)
    }
}

/// Parallel-axis term for moving a tensor of `mass` by `d`: `m (d·d I - d⊗d)`.
fn parallel_axis(mass: f32, d: Vector3) -> Mat3 {
    Mat3::from_diagonal(Vector3::new(
        d.length_sq(),
        d.length_sq(),
        d.length_sq(),
    ))
    .sub(&Mat3::outer(d, d))
    .scaled(mass)
}

/// Volume integrals of a closed, outward-wound triangle mesh, by summing signed
/// tetrahedra from the origin (Blow & Binstock).
///
/// Accumulates the covariance `∫ p ⊗ p dV`, then converts to an inertia tensor
/// with `I = tr(C) * 1 - C` after shifting to the centre of mass.
fn mass_properties_from_triangles(
    triangles: impl Iterator<Item = [Vector3; 3]>,
    density: f32,
) -> MassProperties {
    // Covariance of the canonical tetrahedron (0, e0, e1, e2).
    const CANON: Mat3 = Mat3 {
        m: [
            2.0 / 120.0,
            1.0 / 120.0,
            1.0 / 120.0,
            1.0 / 120.0,
            2.0 / 120.0,
            1.0 / 120.0,
            1.0 / 120.0,
            1.0 / 120.0,
            2.0 / 120.0,
        ],
    };

    let mut volume = 0.0f32;
    let mut weighted_com = Vector3::ZERO;
    let mut covariance = Mat3::ZERO;

    for [a, b, c] in triangles {
        // Columns of the map taking the canonical tet onto this one.
        let map = Mat3 {
            m: [a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z],
        };
        let det = map.determinant();
        let tet_volume = det / 6.0;
        if tet_volume == 0.0 {
            continue;
        }
        volume += tet_volume;
        weighted_com = weighted_com + (a + b + c) * (0.25 * tet_volume);
        covariance = covariance.add(&map.mul(&CANON).mul(&map.transpose()).scaled(det));
    }

    if volume <= 1e-12 {
        return MassProperties::ZERO;
    }

    let com = weighted_com * (1.0 / volume);
    // Shift the covariance to the centre of mass.
    let com_cov = covariance.sub(&Mat3::outer(com, com).scaled(volume));
    let trace = com_cov.m[0] + com_cov.m[4] + com_cov.m[8];
    let inertia = Mat3::from_diagonal(Vector3::new(trace, trace, trace))
        .sub(&com_cov)
        .scaled(density);

    MassProperties::new(density * volume, com, inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn ball_mass_properties_match_the_textbook() {
        let mp = Shape::ball(2.0).mass_properties(3.0);
        let m = 3.0 * (4.0 / 3.0) * PI * 8.0;
        assert!(approx(mp.mass, m, 1e-4));
        assert!(approx(mp.inertia.m[0], 0.4 * m * 4.0, 1e-4));
        assert_eq!(mp.center_of_mass, Vector3::ZERO);
    }

    #[test]
    fn cuboid_mass_properties_match_the_textbook() {
        let mp = Shape::cuboid(1.0, 2.0, 3.0).mass_properties(2.0);
        let m = 2.0 * 8.0 * 6.0;
        assert!(approx(mp.mass, m, 1e-4));
        assert!(approx(mp.inertia.m[0], m / 3.0 * (4.0 + 9.0), 1e-4));
        assert!(approx(mp.inertia.m[4], m / 3.0 * (1.0 + 9.0), 1e-4));
        assert!(approx(mp.inertia.m[8], m / 3.0 * (1.0 + 4.0), 1e-4));
    }

    #[test]
    fn cuboid_from_size_halves_the_dimensions() {
        let a = Shape::cuboid_from_size(2.0, 4.0, 6.0);
        let b = Shape::cuboid(1.0, 2.0, 3.0);
        assert!(approx(a.mass_properties(1.0).mass, b.mass_properties(1.0).mass, 1e-6));
    }

    #[test]
    fn a_zero_length_capsule_is_a_sphere() {
        let cap = Shape::capsule(0.0, 1.3).mass_properties(4.0);
        let ball = Shape::ball(1.3).mass_properties(4.0);
        assert!(approx(cap.mass, ball.mass, 1e-4));
        assert!(approx(cap.inertia.m[0], ball.inertia.m[0], 1e-3));
        assert!(approx(cap.inertia.m[4], ball.inertia.m[4], 1e-3));
    }

    #[test]
    fn hull_of_a_box_matches_the_analytic_box() {
        let h = Vector3::new(0.7, 1.3, 2.1);
        let mut pts = Vec::new();
        for &sx in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                for &sz in &[-1.0f32, 1.0] {
                    pts.push(Vector3::new(h.x * sx, h.y * sy, h.z * sz));
                }
            }
        }
        let hull = Shape::convex_hull(&pts).unwrap().mass_properties(2.5);
        let cube = Shape::cuboid(h.x, h.y, h.z).mass_properties(2.5);
        assert!(approx(hull.mass, cube.mass, 1e-3), "{} vs {}", hull.mass, cube.mass);
        for i in [0, 4, 8] {
            assert!(
                approx(hull.inertia.m[i], cube.inertia.m[i], 1e-2),
                "i={i}: {} vs {}",
                hull.inertia.m[i],
                cube.inertia.m[i]
            );
        }
        assert!(hull.center_of_mass.length() < 1e-4);
    }

    #[test]
    fn mesh_integration_recovers_an_offset_centre_of_mass() {
        // A box whose centre sits at (5, 0, 0).
        let offset = Vector3::new(5.0, 0.0, 0.0);
        let mut pts = Vec::new();
        for &sx in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                for &sz in &[-1.0f32, 1.0] {
                    pts.push(Vector3::new(sx, sy, sz) + offset);
                }
            }
        }
        let mp = Shape::convex_hull(&pts).unwrap().mass_properties(1.0);
        assert!(approx(mp.mass, 8.0, 1e-3));
        assert!((mp.center_of_mass - offset).length() < 1e-3, "{:?}", mp.center_of_mass);
        // Inertia about the COM must equal the centred box's.
        let cube = Shape::cuboid(1.0, 1.0, 1.0).mass_properties(1.0);
        assert!(approx(mp.inertia.m[0], cube.inertia.m[0], 1e-2));
    }

    #[test]
    fn cone_centre_of_mass_is_a_quarter_up_from_the_base() {
        let mp = Shape::cone(2.0, 1.0).mass_properties(1.0);
        assert!(approx(mp.center_of_mass.y, -1.0, 1e-4));
        assert!(approx(mp.mass, PI * 1.0 * 4.0 / 3.0, 1e-4));
    }

    #[test]
    fn compound_of_two_halves_equals_the_whole() {
        let whole = Shape::cuboid(2.0, 1.0, 1.0).mass_properties(1.0);
        let half = Shape::cuboid(1.0, 1.0, 1.0);
        let compound = Shape::compound(vec![
            (Isometry::from_translation(Vector3::new(-1.0, 0.0, 0.0)), half.clone()),
            (Isometry::from_translation(Vector3::new(1.0, 0.0, 0.0)), half),
        ])
        .mass_properties(1.0);
        assert!(approx(compound.mass, whole.mass, 1e-4));
        assert!(compound.center_of_mass.length() < 1e-5);
        for i in [0, 4, 8] {
            assert!(
                approx(compound.inertia.m[i], whole.inertia.m[i], 1e-3),
                "i={i}: {} vs {}",
                compound.inertia.m[i],
                whole.inertia.m[i]
            );
        }
    }

    #[test]
    fn half_space_and_open_mesh_have_no_mass() {
        assert!(Shape::ground().mass_properties(1.0).is_zero());
        assert!(Shape::ground().is_static_only());
        // A single triangle is not a closed solid.
        let open = Shape::trimesh(
            vec![Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)],
            vec![[0, 1, 2]],
        )
        .unwrap();
        assert!(open.mass_properties(1.0).is_zero());
    }

    #[test]
    fn support_points_are_extreme() {
        let s = Shape::cuboid(1.0, 2.0, 3.0);
        assert_eq!(
            s.support_local(Vector3::new(1.0, -1.0, 1.0)).unwrap(),
            Vector3::new(1.0, -2.0, 3.0)
        );
        let c = Shape::capsule(2.0, 0.5);
        let top = c.support_local(Vector3::UP).unwrap();
        assert!(approx(top.y, 2.5, 1e-5));
        let cone = Shape::cone(1.0, 1.0);
        assert_eq!(cone.support_local(Vector3::UP).unwrap(), Vector3::new(0.0, 1.0, 0.0));
        assert!(Shape::ground().support_local(Vector3::UP).is_none());
    }

    #[test]
    fn containment_matches_each_primitive() {
        assert!(Shape::ball(1.0).contains_point_local(Vector3::new(0.5, 0.5, 0.5)));
        assert!(!Shape::ball(1.0).contains_point_local(Vector3::new(0.9, 0.9, 0.0)));
        assert!(Shape::cylinder(1.0, 1.0).contains_point_local(Vector3::new(0.9, 0.9, 0.0)));
        assert!(!Shape::cylinder(1.0, 1.0).contains_point_local(Vector3::new(0.0, 1.1, 0.0)));
        // Cone tapers: a point near the apex must be close to the axis.
        let cone = Shape::cone(1.0, 1.0);
        assert!(cone.contains_point_local(Vector3::new(0.4, -0.5, 0.0)));
        assert!(!cone.contains_point_local(Vector3::new(0.4, 0.5, 0.0)));
        // Half-space is solid below y = 0.
        assert!(Shape::ground().contains_point_local(Vector3::new(9.0, -0.1, 9.0)));
        assert!(!Shape::ground().contains_point_local(Vector3::new(0.0, 0.1, 0.0)));
    }

    #[test]
    fn rotated_aabbs_enclose_the_shape() {
        let iso = Isometry::new(
            Vector3::new(1.0, 2.0, 3.0),
            threers::math::Quaternion::from_euler_xyz(0.4, 0.9, -0.3).normalize(),
        );
        for s in [
            Shape::ball(0.75),
            Shape::cuboid(1.0, 0.5, 2.0),
            Shape::capsule(1.0, 0.4),
            Shape::cylinder(1.0, 0.6),
        ] {
            let aabb = s.compute_aabb(&iso);
            // Sample the support in many directions; every point must be inside.
            for i in 0..64 {
                let a = i as f32 * 0.31;
                let dir = Vector3::new(a.sin(), (a * 1.7).cos(), (a * 0.6).sin());
                let p = iso.transform_point(
                    s.support_local(iso.inverse_transform_vector(dir)).unwrap(),
                );
                assert!(
                    p.x >= aabb.min.x - 1e-4
                        && p.x <= aabb.max.x + 1e-4
                        && p.y >= aabb.min.y - 1e-4
                        && p.y <= aabb.max.y + 1e-4
                        && p.z >= aabb.min.z - 1e-4
                        && p.z <= aabb.max.z + 1e-4,
                    "{p:?} escaped {aabb:?}"
                );
            }
        }
    }
}
