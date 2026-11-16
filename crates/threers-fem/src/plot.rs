//! Results as renderable geometry: a mesh's boundary, optionally deformed,
//! coloured by a field.

use crate::mesh::TetMesh;
use crate::vec3::*;
use threers::core::{BufferAttribute, BufferGeometry};
use threers::math::ColorRamp;

/// The boundary of `mesh` with each surface vertex at `positions[node]`
/// and coloured by `values[node]` through `ramp`, mapped linearly from
/// `range` (or the values' own extent when `None`). Positions and values are
/// per mesh node, as every solution in this crate reports them.
///
/// Draw it with `vertex_colors` on the material.
pub fn field_geometry(
    mesh: &TetMesh,
    positions: &[V3],
    values: &[f64],
    ramp: &ColorRamp,
    range: Option<(f64, f64)>,
) -> BufferGeometry {
    assert_eq!(positions.len(), mesh.nodes.len());
    assert_eq!(values.len(), mesh.nodes.len());
    let (surface, back) = mesh.boundary_surface();
    let (lo, hi) = range.unwrap_or_else(|| {
        back.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &n| {
            let v = values[n as usize];
            (lo.min(v), hi.max(v))
        })
    });
    let span = if hi > lo { hi - lo } else { 1.0 };
    let mut moved = surface.clone();
    for (v, &n) in moved.vertices.iter_mut().zip(&back) {
        *v = positions[n as usize];
    }
    let mut g = moved.to_geometry();
    let colors: Vec<f32> = back
        .iter()
        .flat_map(|&n| {
            let c = ramp.at(((values[n as usize] - lo) / span) as f32);
            [c.r, c.g, c.b]
        })
        .collect();
    g.set_attribute("color", BufferAttribute::new(colors, 3));
    g
}

/// The undeformed boundary of `mesh`, uncoloured — to look at a mesh
/// before solving on it.
pub fn mesh_geometry(mesh: &TetMesh) -> BufferGeometry {
    mesh.boundary_surface().0.to_geometry()
}

impl crate::fem::ElasticSolution {
    /// The deformed part, `exaggerate` times the true displacement, coloured
    /// by von Mises stress.
    pub fn von_mises_geometry(&self, mesh: &TetMesh, exaggerate: f64, ramp: &ColorRamp) -> BufferGeometry {
        field_geometry(mesh, &self.deformed_nodes(exaggerate), &self.von_mises, ramp, None)
    }
}

impl crate::fem::HeatSolution {
    /// The part coloured by temperature.
    pub fn temperature_geometry(&self, mesh: &TetMesh, ramp: &ColorRamp) -> BufferGeometry {
        field_geometry(mesh, &mesh.nodes, &self.temperature, ramp, None)
    }
}

impl crate::fem::Mode {
    /// The part displaced by this mode shape at `amplitude` (in length units;
    /// the shape's largest displacement is 1), coloured by displacement.
    /// Animate `amplitude · sin(ωt)` to watch it vibrate.
    pub fn geometry(&self, mesh: &TetMesh, amplitude: f64, ramp: &ColorRamp) -> BufferGeometry {
        let pos: Vec<V3> = mesh
            .nodes
            .iter()
            .zip(&self.shape)
            .map(|(p, d)| add(*p, scale(*d, amplitude)))
            .collect();
        let mag: Vec<f64> = self.shape.iter().map(|d| norm(*d)).collect();
        field_geometry(mesh, &pos, &mag, ramp, Some((0.0, 1.0)))
    }
}
