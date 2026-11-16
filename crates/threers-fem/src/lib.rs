//! Finite and boundary element analysis for [`threers`](https://docs.rs/threers).
//!
//! - **Mesh** a closed surface — any `BufferGeometry`, or a signed distance
//!   function — into tetrahedra: [`TetMesh::from_geometry`],
//!   [`TetMesh::from_sdf`]. See [`mesher`] for how and what it gives up.
//! - **FEM** on the tetrahedra: [`Elastic`] for displacement, stress and
//!   reactions, [`Elastic::modes`] for natural frequencies, [`Heat`] for steady
//!   conduction. See [`fem`].
//! - **BEM** on the surface alone: [`bem::Laplace`] for potential problems —
//!   electrostatics, capacitance, potential flow, steady conduction in a
//!   homogeneous body — and [`bem::Helmholtz`] for acoustic radiation and
//!   scattering into open space. See [`bem`].
//!
//! ```
//! use threers_fem::{Elastic, Material, TetMesh};
//!
//! // A steel cantilever, 100 × 10 × 10 mm, in millimetres and newtons
//! // (so stresses come out in MPa).
//! let mesh = TetMesh::box_grid([0.0; 3], [100.0, 10.0, 10.0], [20, 2, 2]);
//! let s = Elastic::new(&mesh, Material::new(200e3, 0.3))
//!     .fix(|p| p[0] < 1e-9)
//!     .force(|p| p[0] > 100.0 - 1e-9, [0.0, -100.0, 0.0])
//!     .solve()
//!     .unwrap();
//! // Beam theory: δ = FL³ / 3EI = 0.2 mm.
//! assert!((s.max_displacement() - 0.2).abs() < 0.01);
//! ```
//!
//! # Which method
//!
//! FEM meshes the volume and handles anything: any material, internal
//! sources, a part that is part steel and part rubber. BEM meshes only the
//! surface and is exact away from it, which makes it the method for fields
//! in an *unbounded* region — the sound a part radiates, the electric field
//! around a conductor — where a volume mesh would have to stop somewhere. Its
//! matrices are dense, so it is for surfaces of up to a few thousand
//! triangles.
//!
//! # Feature flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `parallel` | Assembly, BEM matrices and field evaluation across threads via rayon. Native only; a no-op on wasm32. |
//! | `wasm` | JavaScript bindings through wasm-bindgen ([`wasm`], wasm32 only), over the [`flat`] API. |
//!
//! Everything else is pure Rust with no dependencies beyond `threers`
//! (without its renderer), and runs in the browser.

pub mod bem;
pub mod complex;
pub mod dense;
pub mod fem;
pub mod flat;
pub mod mesh;
pub mod mesher;
pub mod plot;
pub mod sparse;
pub mod surface;
pub mod vec3;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;

pub use bem::{Helmholtz, Laplace};
pub use complex::Complex;
pub use fem::{Elastic, ElasticSolution, Heat, HeatSolution, Material, Mode, Modes, Order};
pub use mesh::{MeshQuality, TetMesh};
pub use sparse::LinearSolver;
pub use surface::{SignedDistance, TriSurface};

/// What can go wrong setting up or solving a problem.
#[derive(Clone, Debug, PartialEq)]
pub enum FemError {
    /// A parameter or input out of range; the message says which.
    InvalidInput(&'static str),
    /// A surface that is not a closed, consistently wound triangle mesh.
    InvalidSurface(&'static str),
    /// A support or load whose region selected no nodes or faces; names
    /// which kind.
    EmptyRegion(&'static str),
    /// A singular dense system.
    Singular,
    /// A stiffness that is not positive definite: almost always supports
    /// that leave the part free to move as a rigid body.
    NotPositiveDefinite,
    /// An iterative solve that ran out of iterations.
    NoConvergence { iterations: usize },
}

impl std::fmt::Display for FemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::InvalidSurface(m) => write!(f, "invalid surface: {m}"),
            Self::EmptyRegion(k) => write!(f, "the {k} region selects nothing on the mesh"),
            Self::Singular => write!(f, "the system is singular"),
            Self::NotPositiveDefinite => write!(
                f,
                "the stiffness is not positive definite — are the supports enough to stop rigid motion?"
            ),
            Self::NoConvergence { iterations } => {
                write!(f, "no convergence after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for FemError {}

pub mod prelude {
    pub use crate::bem::{Helmholtz, Laplace, Potential};
    pub use crate::complex::Complex;
    pub use crate::fem::{Elastic, Heat, Material, Order};
    pub use crate::mesh::TetMesh;
    pub use crate::sparse::LinearSolver;
    pub use crate::surface::TriSurface;
    pub use crate::FemError;
}
