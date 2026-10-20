//! CSG brush: geometry + world matrix + BVH / half-edges for boolean ops.
//!
//! three-bvh-csg 0.0.16 `Brush`.

use crate::core::BufferGeometry;
use crate::math::Matrix4;
use crate::mesh_bvh::{BuildOptions, MeshBvh};

use super::geometry_prep::{build_group_indices, ensure_index, ensure_non_indexed};
use super::half_edge_map::HalfEdgeMap;
use super::js_topology::JsMatrix4;

/// A mesh operand for CSG evaluation (three-bvh-csg's `Brush`): geometry in
/// its own frame, and its transforms as three.js holds them — `matrix` (local)
/// and `matrix_world`, column-major f64. They stay f64 throughout because
/// three-bvh-csg composes and inverts them in JavaScript's doubles, and the
/// classification of a vertex lying on another brush's face turns on the
/// last bit.
#[derive(Debug)]
pub struct CsgBrush {
    pub geometry: BufferGeometry,
    /// three.js `matrix`: this brush in its parent's frame.
    pub matrix: [f64; 16],
    /// three.js `matrixWorld`: this brush in world space.
    pub matrix_world: [f64; 16],
    previous_matrix: [f64; 16],
    pub(crate) half_edges: Option<HalfEdgeMap>,
    pub(crate) group_indices: Vec<u16>,
}

/// The column-major identity.
pub const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

impl CsgBrush {
    pub fn new(geometry: BufferGeometry) -> Self {
        Self {
            geometry,
            matrix: IDENTITY,
            matrix_world: IDENTITY,
            // Never equal to a real matrix, so a new brush is dirty, as in
            // three-bvh-csg.
            previous_matrix: [0.0; 16],
            half_edges: None,
            group_indices: Vec::new(),
        }
    }

    /// three.js `position.set(x, y, z)` on an unrotated, unscaled brush.
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.matrix = IDENTITY;
        self.matrix[12] = x;
        self.matrix[13] = y;
        self.matrix[14] = z;
    }

    /// Set `matrix_world` (and `matrix`) as three.js computed it — for a
    /// brush whose scene graph lives elsewhere, such as JavaScript.
    pub fn set_matrix_world(&mut self, elements: [f64; 16]) {
        self.matrix = elements;
        self.matrix_world = elements;
    }

    /// three.js `updateMatrixWorld`: `parent · matrix`, or `matrix` at the root.
    pub fn update_matrix_world(&mut self, parent: Option<&[f64; 16]>) {
        self.matrix_world = match parent {
            Some(p) => {
                JsMatrix4 { elements: *p }
                    .multiply(JsMatrix4 {
                        elements: self.matrix,
                    })
                    .elements
            }
            None => self.matrix,
        };
    }

    pub(crate) fn js_matrix_world(&self) -> JsMatrix4 {
        JsMatrix4 {
            elements: self.matrix_world,
        }
    }

    /// `matrix_world` in f32, for the BVH traversal's box tests.
    pub(crate) fn matrix_world_f32(&self) -> Matrix4 {
        Matrix4 {
            elements: self.matrix_world.map(|e| e as f32),
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.matrix != self.previous_matrix
    }

    pub fn mark_updated(&mut self) {
        self.previous_matrix = self.matrix;
    }

    /// Build index, BVH, and half-edge map (required before evaluate).
    pub fn prepare_geometry(&mut self) {
        // Primitives arrive as non-indexed soup in JS (`geometryToBufferGeometry` →
        // `toNonIndexed`); mirror that before `ensureIndex`.
        ensure_non_indexed(&mut self.geometry);
        ensure_index(&mut self.geometry);
        if self.geometry.bounds_tree.is_none() {
            let opts = BuildOptions {
                max_leaf_tris: 3,
                indirect: true,
                ..Default::default()
            };
            self.geometry.compute_bounds_tree(opts);
        }
        self.half_edges = Some(HalfEdgeMap::from_geometry(&self.geometry));
        self.group_indices = build_group_indices(&self.geometry);
    }

    pub fn bounds_tree(&self) -> std::sync::Arc<MeshBvh> {
        self.geometry
            .bounds_tree
            .as_ref()
            .expect("bounds tree")
            .clone()
    }
}
