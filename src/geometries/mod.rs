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
pub mod frenet;
mod kirigami;
mod lathe_geometry;
pub mod lattice;
#[cfg(feature = "nurbs")]
mod nurbs_geometry;
mod parametric_geometry;
mod plane_geometry;
mod polygon_geometry;
pub(crate) mod polygons;
mod polyhedron_geometry;
mod quad_geometry;
#[cfg(test)]
mod r165_parity_tests;
mod ring_geometry;
mod rounded_box_geometry;
mod shape_geometry;
mod sphere_geometry;
pub mod subdivision;
mod teapot_data;
mod teapot_geometry;
mod text_geometry;
mod torus_geometry;
mod torus_knot_geometry;
mod tube_geometry;

pub use box_geometry::BoxGeometry;
pub use box_line::BoxLineGeometry;
pub use capsule_geometry::CapsuleGeometry;
pub use circle_geometry::CircleGeometry;
pub use cone_geometry::ConeGeometry;
pub use convex_geometry::{ConvexGeometry, ConvexHull, HullFace};
pub use cylinder_geometry::CylinderGeometry;
pub use decal_geometry::{DecalGeometry, EulerOrder};
pub use edges_geometry::{EdgesGeometry, WireframeGeometry};
pub use extrude_geometry::{
    ExtrudeBuffers, ExtrudeFace, ExtrudeGeometry, ExtrudeOptions, ExtrudePath,
};
pub use kirigami::{
    KirigamiAssembly, KirigamiCell, KirigamiCoreLattice, KirigamiCrease, KirigamiCreaseKind,
    KirigamiExpandedMiura, KirigamiFace, KirigamiFaceKind, KirigamiMesh, KirigamiNet,
    KirigamiNetPanel, KirigamiPreset, KIRIGAMI_NET_VARIANT,
};
pub use lathe_geometry::LatheGeometry;
pub use lattice::{
    ChiralRule, Conductivity, Conform, Cuboct, CuboctAssembly, CuboctAssemblyPlan,
    CuboctAssemblyStep, CuboctFrame, CuboctJoint, CuboctJointKind, Field, FrameMaterial,
    FrameResponse, Hand, Infill, Lattice, LatticeGeometry, LatticeKind, LatticeMetrics,
    LatticeStyle, Region, SeedLayout, Segment, SolidMaterial, Solver, Stiffness, Stochastic, Strength, Strut,
    Tpms,
};
#[cfg(feature = "nurbs")]
pub use nurbs_geometry::NurbsGeometry;
pub use parametric_geometry::ParametricGeometry;
pub use plane_geometry::PlaneGeometry;
pub use polygon_geometry::PolygonGeometry;
pub use polyhedron_geometry::{
    DodecahedronGeometry, IcosahedronGeometry, OctahedronGeometry, PolyhedronGeometry,
    TetrahedronGeometry,
};
pub use quad_geometry::QuadGeometry;
pub use ring_geometry::RingGeometry;
pub use rounded_box_geometry::{three_box_geometry, RoundedBoxGeometry};
pub use shape_geometry::ShapeGeometry;
pub use sphere_geometry::SphereGeometry;
pub use subdivision::SubdivisionOptions;
pub use teapot_geometry::{TeapotGeometry, TeapotOptions};
pub use text_geometry::{Glyph, TextGeometry, TextOptions};
pub use torus_geometry::TorusGeometry;
pub use torus_knot_geometry::TorusKnotGeometry;
pub use tube_geometry::TubeGeometry;

/// The number a three.js caller wrote for `v`: the shortest decimal that
/// rounds to this `f32`, as an `f64`. three.js builds geometry in f64 from the
/// literal it was given (`0.4`) and rounds once into a `Float32Array`;
/// widening the `f32` instead (`0.4000000059604645`) lands vertices on the
/// other side of an axis they sit a hair from, which decides ray hits and CSG
/// splits there.
pub(crate) fn js_number(v: f32) -> f64 {
    format!("{v}").parse().unwrap_or(v as f64)
}

/// [`js_number`] for an angle: an `f32` that is π, 2π, π/2 or another
/// quarter-turn multiple stands for three.js's f64 constant (`Math.PI * 2`
/// is not `6.2831855`).
pub(crate) fn js_angle(v: f32) -> f64 {
    for k in 1..=8 {
        let a = std::f64::consts::FRAC_PI_4 * k as f64;
        if v == a as f32 {
            return a;
        }
    }
    js_number(v)
}

/// Generator buffers as three.js fills them: numbers computed in f64,
/// rounded once into the `Float32Array` (`Float32BufferAttribute`).
#[derive(Default)]
pub(crate) struct JsBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub indices: Vec<u32>,
    pub groups: Vec<(u32, u32, u32)>,
}

impl JsBuffers {
    pub fn vertex(&mut self, x: f64, y: f64, z: f64) {
        self.positions
            .extend_from_slice(&[x as f32, y as f32, z as f32]);
    }
    pub fn normal(&mut self, x: f64, y: f64, z: f64) {
        self.normals
            .extend_from_slice(&[x as f32, y as f32, z as f32]);
    }
    /// `normal.set(x, y, z).normalize()`: three.js scales by the reciprocal
    /// of the length (`length() || 1`).
    pub fn normal_normalized(&mut self, x: f64, y: f64, z: f64) {
        let len = (x * x + y * y + z * z).sqrt();
        let inv = 1.0 / if len == 0.0 { 1.0 } else { len };
        self.normal(x * inv, y * inv, z * inv);
    }
    pub fn uv(&mut self, u: f64, v: f64) {
        self.uvs.extend_from_slice(&[u as f32, v as f32]);
    }
    pub fn vertex_count(&self) -> u32 {
        (self.positions.len() / 3) as u32
    }
    /// The geometry, with the index (if any) and groups.
    pub fn into_geometry(self) -> crate::core::BufferGeometry {
        use crate::core::{BufferAttribute, BufferGeometry, GeometryGroup};
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(self.positions, 3));
        g.set_attribute("normal", BufferAttribute::new(self.normals, 3));
        if !self.uvs.is_empty() {
            g.set_attribute("uv", BufferAttribute::new(self.uvs, 2));
        }
        if !self.indices.is_empty() {
            g.set_index(self.indices);
        }
        g.groups = self
            .groups
            .into_iter()
            .map(|(start, count, material_index)| GeometryGroup {
                start,
                count,
                material_index,
            })
            .collect();
        g
    }
}
