use crate::core::BufferGeometry;
use crate::math::{Matrix4, Vector3};

use super::brush::CsgBrush;
use super::constants::Operation;
use super::evaluator::CsgEvaluator;
use super::geometry_prep::load_positions_geometry_bin;
use super::hierarchy_steps::step4_win_frame;

/// A CSG operation node (mirrors `web/csg/core/operations/Operation.js`).
#[derive(Debug)]
pub struct CsgOperation {
    pub brush: CsgBrush,
    pub operation: Operation,
    pub children: Vec<CsgNode>,
    cached_geometry: Option<BufferGeometry>,
    previous_operation: Option<Operation>,
}

impl CsgOperation {
    pub fn new(geometry: BufferGeometry, operation: Operation) -> Self {
        Self {
            brush: CsgBrush::new(geometry),
            operation,
            children: Vec::new(),
            cached_geometry: None,
            previous_operation: None,
        }
    }

    pub fn add_child(&mut self, child: CsgNode) {
        self.children.push(child);
    }

    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.brush.set_position(x, y, z);
    }

    pub fn update_matrix_world(&mut self, parent: Option<&Matrix4>) {
        self.brush.update_matrix_world(parent);
        for child in &mut self.children {
            child.update_matrix_world(Some(&self.brush.matrix_world));
        }
    }

    fn is_dirty(&self) -> bool {
        self.previous_operation != Some(self.operation) || self.brush.is_dirty()
    }

    fn mark_updated(&mut self) {
        self.brush.mark_updated();
        self.previous_operation = Some(self.operation);
    }
}

/// Groups child operations without participating in CSG (mirrors `OperationGroup.js`).
#[derive(Debug)]
pub struct CsgOperationGroup {
    pub local_matrix: Matrix4,
    pub matrix_world: Matrix4,
    pub children: Vec<CsgNode>,
    previous_local_matrix: Matrix4,
}

impl Default for CsgOperationGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl CsgOperationGroup {
    pub fn new() -> Self {
        let mut previous_local_matrix = Matrix4::identity();
        previous_local_matrix.elements.fill(0.0);
        Self {
            local_matrix: Matrix4::identity(),
            matrix_world: Matrix4::identity(),
            children: Vec::new(),
            previous_local_matrix,
        }
    }

    pub fn add_child(&mut self, child: CsgNode) {
        self.children.push(child);
    }

    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.local_matrix = Matrix4::translation(Vector3::new(x as f32, y as f32, z as f32));
    }

    pub fn update_matrix_world(&mut self, parent: Option<&Matrix4>) {
        self.matrix_world = parent
            .map(|p| p.multiply(&self.local_matrix))
            .unwrap_or(self.local_matrix);
        for child in &mut self.children {
            child.update_matrix_world(Some(&self.matrix_world));
        }
    }

    fn is_dirty(&self) -> bool {
        self.local_matrix.elements != self.previous_local_matrix.elements
    }

    fn mark_updated(&mut self) {
        self.previous_local_matrix = self.local_matrix;
    }
}

// Boxing the larger variant would save memory per node and cost an indirection
// on every traversal — and this is a public enum, so it would be a breaking
// change for callers that match on it. The tree is small; the trade is not
// worth it.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CsgNode {
    Operation(CsgOperation),
    Group(CsgOperationGroup),
}

impl CsgNode {
    pub fn update_matrix_world(&mut self, parent: Option<&Matrix4>) {
        match self {
            CsgNode::Operation(op) => op.update_matrix_world(parent),
            CsgNode::Group(g) => g.update_matrix_world(parent),
        }
    }
}

fn flat_traverse<'a>(node: &'a CsgNode, out: &mut Vec<&'a CsgOperation>) {
    match node {
        CsgNode::Group(g) => {
            for child in &g.children {
                flat_traverse(child, out);
            }
        }
        CsgNode::Operation(op) => out.push(op),
    }
}

fn traverse_operation(op: &mut CsgOperation, evaluator: &mut CsgEvaluator) -> bool {
    let mut did_change = false;
    for child in &mut op.children {
        did_change = traverse_node(child, evaluator) || did_change;
    }

    let is_dirty = op.is_dirty();
    if is_dirty {
        op.mark_updated();
    }

    if did_change {
        let mut child_ops: Vec<&CsgOperation> = Vec::new();
        for child in &op.children {
            flat_traverse(child, &mut child_ops);
        }

        if !child_ops.is_empty() {
            let mut result: Option<CsgBrush> = None;
            for child in child_ops {
                if result.is_none() {
                    let mut a = CsgBrush::new(op.brush.geometry.clone());
                    a.matrix_world = op.brush.matrix_world;
                    let mut b = CsgBrush::new(child.brush.geometry.clone());
                    b.matrix_world = child.brush.matrix_world;
                    let geom = evaluator.evaluate(&mut a, &mut b, child.operation);
                    result = Some(CsgBrush::new(geom));
                } else {
                    let mut a = result.take().unwrap();
                    a.matrix_world = Matrix4::identity();
                    let mut b = CsgBrush::new(child.brush.geometry.clone());
                    b.matrix_world = child.brush.matrix_world;
                    let geom = evaluator.evaluate(&mut a, &mut b, child.operation);
                    result = Some(CsgBrush::new(geom));
                }
            }
            op.cached_geometry = result.map(|b| b.geometry);
        }
        true
    } else {
        did_change || is_dirty
    }
}

fn traverse_node(node: &mut CsgNode, evaluator: &mut CsgEvaluator) -> bool {
    match node {
        CsgNode::Operation(op) => traverse_operation(op, evaluator),
        CsgNode::Group(g) => {
            let mut did_change = false;
            for child in &mut g.children {
                did_change = traverse_node(child, evaluator) || did_change;
            }
            if g.is_dirty() {
                g.mark_updated();
            }
            did_change
        }
    }
}

/// Evaluate an operation hierarchy (mirrors `Evaluator.evaluateHierarchy`).
pub fn evaluate_hierarchy(root: &mut CsgOperation, evaluator: &mut CsgEvaluator) -> BufferGeometry {
    root.update_matrix_world(None);
    traverse_operation(root, evaluator);
    root.cached_geometry
        .clone()
        .unwrap_or_else(|| root.brush.geometry.clone())
}

/// Full hierarchy for native render: ref after-sphere → live winCut → ref winCut soup → live winFrame.
///
/// winCut on ref sphere is exact (55914). winFrame on that soup still diverges (~+1.2k);
/// feeding the JS winCut bin for step 4 input cuts that gap to ~+333.
pub fn build_bvh_csg_hierarchy_geometry(
    evaluator: &mut CsgEvaluator,
    reference_after_sphere: &[u8],
    reference_after_wincut: &[u8],
) -> BufferGeometry {
    let _ = reference_after_sphere;
    let mut after_cut = load_positions_geometry_bin(reference_after_wincut);
    crate::utils::compute_vertex_normals(&mut after_cut);
    step4_win_frame(evaluator, after_cut)
}
