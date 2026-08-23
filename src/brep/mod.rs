//! Analytic surface provenance — every triangle knows what it came from.
//!
//! Stage 1 of [`docs/brep-nurbs-plan.md`](../../docs/brep-nurbs-plan.md), behind
//! the `brep` feature (which implies `nurbs`).
//!
//! # The problem this solves
//!
//! `sphere(r)` becomes a 32-facet mesh before the first boolean runs, and the
//! sphere is gone. Everything downstream then works from triangles: normals are
//! averaged from the faces we just invented, UVs come from whatever the mesher
//! guessed, resolution is frozen at whatever `$fn` was set before the boolean,
//! and the CSG kernel recovers "one flat face" by *hashing a quantized plane*
//! out of the triangles (`exact_csg::plane_key`) — which works for planes and
//! nothing else.
//!
//! This module keeps the surface. A [`SurfaceTable`] rides alongside a
//! `BufferGeometry` mapping each triangle to the [`Surface`] it was sampled
//! from, and four things follow directly:
//!
//! * **Exact normals** — from `Sᵤ × Sᵥ`, not from area-weighted face averages.
//! * **Exact UVs** — from the surface's own parameterization, which is what
//!   makes pole and seam behaviour predictable instead of per-primitive folklore.
//! * **Re-tessellation** — boolean at a coarse resolution, render at a fine one.
//! * **Real face identity** for Stage 2, so a coaxial-cylinder pair is decided
//!   by comparing two structs instead of by numeric tri-tri intersection.
//!
//! # Provenance is optional, always
//!
//! `BufferGeometry::surfaces` is an `Option`, and every consumer treats `None`
//! as "fall back to the mesh". Nothing in this module is load-bearing: a
//! transform that cannot be represented drops the table, an edit to the
//! positions invalidates it, and the result is a build that behaves exactly as
//! it did before the feature existed. That is the plan's safety property, and it
//! is what makes it safe to attach provenance to primitives by default.

pub mod body;
#[cfg(feature = "brep-kernel")]
pub mod boolean;
pub mod curve3d;
#[cfg(feature = "brep-csg")]
pub mod intersect;
/// Subdividing a face's parameter region by the curves drawn on it — the
/// planar arrangement a face split by *open* curves needs.
#[cfg(feature = "brep-kernel")]
pub mod planar;
pub mod primitives;
pub mod provenance;
pub mod retessellate;
pub mod stitch;
pub mod surface;
pub mod table;

pub use body::{Body, BodyReport, Defect, Face, Shell, TrimLoop};
#[cfg(feature = "brep-kernel")]
pub use boolean::{BooleanOp, Declined};
pub use curve3d::Curve3d;
#[cfg(feature = "brep-csg")]
pub use intersect::{march, ssi, SsiResult};
pub use provenance::{
    exact_normals, exact_uvs, exact_uvs_split_seams, recover_planar_surfaces, recover_surfaces,
    FitReport, UvReport,
};
pub use retessellate::{retessellate, RetessellationReport};
pub use stitch::{densify_to, patch_boundaries, Boundary};
pub use surface::Surface;
pub use table::{triangle_count, triangle_indices, triangle_vertices, SurfaceTable};
