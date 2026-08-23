//! Built-in geometry generators, mirroring three.js's geometry module.

mod box_geometry;
mod box_line;
mod capsule_geometry;
mod circle_geometry;
mod cone_geometry;
mod convex_geometry;
mod cylinder_geometry;
mod decal_geometry;
mod edges_geometry;
mod extrude_geometry;
mod kirigami;
mod lathe_geometry;
pub mod lattice;
#[cfg(feature = "nurbs")]
mod nurbs_geometry;
mod parametric_geometry;
mod plane_geometry;
mod polyhedron_geometry;
mod ring_geometry;
mod sphere_geometry;
mod text_geometry;
mod torus_geometry;
mod torus_knot_geometry;
mod tube_geometry;

pub use box_geometry::BoxGeometry;
pub use box_line::BoxLineGeometry;
pub use capsule_geometry::CapsuleGeometry;
pub use circle_geometry::CircleGeometry;
pub use cone_geometry::ConeGeometry;
pub use convex_geometry::ConvexGeometry;
pub use cylinder_geometry::CylinderGeometry;
pub use decal_geometry::DecalGeometry;
pub use edges_geometry::{EdgesGeometry, WireframeGeometry};
pub use extrude_geometry::ExtrudeGeometry;
pub use kirigami::{
    KirigamiAssembly, KirigamiCell, KirigamiCoreLattice, KirigamiCrease, KirigamiCreaseKind,
    KirigamiExpandedMiura, KirigamiFace, KirigamiFaceKind, KirigamiMesh, KirigamiNet,
    KirigamiNetPanel, KirigamiPreset, KIRIGAMI_NET_VARIANT,
};
pub use lathe_geometry::LatheGeometry;
pub use lattice::{
    ChiralRule, Cuboct, CuboctAssembly, CuboctAssemblyPlan, CuboctAssemblyStep, CuboctFrame,
    CuboctJoint, CuboctJointKind, FrameMaterial, FrameResponse, Hand, Infill, Lattice,
    LatticeGeometry, LatticeKind, LatticeStyle, Region, Segment, Strut, Tpms,
};
#[cfg(feature = "nurbs")]
pub use nurbs_geometry::NurbsGeometry;
pub use parametric_geometry::ParametricGeometry;
pub use plane_geometry::PlaneGeometry;
pub use polyhedron_geometry::{
    DodecahedronGeometry, IcosahedronGeometry, OctahedronGeometry, PolyhedronGeometry,
    TetrahedronGeometry,
};
pub use ring_geometry::RingGeometry;
pub use sphere_geometry::SphereGeometry;
pub use text_geometry::{Glyph, TextGeometry};
pub use torus_geometry::TorusGeometry;
pub use torus_knot_geometry::TorusKnotGeometry;
pub use tube_geometry::TubeGeometry;
