//! three-bvh-csg 0.0.16's `Operation`, `OperationGroup` and
//! `Evaluator.evaluateHierarchy`: a tree of brushes, each applied to its
//! parent with its operation, re-evaluated only where something changed.

use crate::core::BufferGeometry;

use super::brush::{CsgBrush, IDENTITY};
use super::constants::Operation;
use super::evaluator::CsgEvaluator;
use super::js_topology::JsMatrix4;

/// three-bvh-csg `Operation`: a brush applied to its parent with
/// `operation`, whose own children apply to it in turn.
#[derive(Debug)]
pub struct CsgOperation {
    pub brush: CsgBrush,
    pub operation: Operation,
    pub children: Vec<CsgNode>,
    /// This operation with its children applied, from the last evaluation
    /// that changed it (three-bvh-csg's `_cachedGeometry`).
    cached_geometry: BufferGeometry,
    previous_operation: Option<Operation>,
}

impl CsgOperation {
    pub fn new(geometry: BufferGeometry, operation: Operation) -> Self {
        Self {
            brush: CsgBrush::new(geometry),
            operation,
            children: Vec::new(),
            cached_geometry: BufferGeometry::new(),
            previous_operation: None,
        }
    }

    pub fn add_child(&mut self, child: CsgNode) {
        self.children.push(child);
    }

    /// three.js `position.set(x, y, z)`.
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.brush.set_position(x, y, z);
    }

    pub fn update_matrix_world(&mut self, parent: Option<&[f64; 16]>) {
        self.brush.update_matrix_world(parent);
        let world = self.brush.matrix_world;
        for child in &mut self.children {
            child.update_matrix_world(Some(&world));
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

/// three-bvh-csg `OperationGroup`: a transform its children inherit, taking
/// no part in the boolean itself.
#[derive(Debug)]
pub struct CsgOperationGroup {
    /// three.js `matrix`.
    pub matrix: [f64; 16],
    /// three.js `matrixWorld`.
    pub matrix_world: [f64; 16],
    pub children: Vec<CsgNode>,
    previous_matrix: [f64; 16],
}

impl Default for CsgOperationGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl CsgOperationGroup {
    pub fn new() -> Self {
        Self {
            matrix: IDENTITY,
            matrix_world: IDENTITY,
            children: Vec::new(),
            previous_matrix: [0.0; 16],
        }
    }

    pub fn add_child(&mut self, child: CsgNode) {
        self.children.push(child);
    }

    /// three.js `position.set(x, y, z)`.
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.matrix = IDENTITY;
        self.matrix[12] = x;
        self.matrix[13] = y;
        self.matrix[14] = z;
    }

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
        let world = self.matrix_world;
        for child in &mut self.children {
            child.update_matrix_world(Some(&world));
        }
    }

    fn is_dirty(&self) -> bool {
        self.matrix != self.previous_matrix
    }

    fn mark_updated(&mut self) {
        self.previous_matrix = self.matrix;
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
    pub fn update_matrix_world(&mut self, parent: Option<&[f64; 16]>) {
        match self {
            CsgNode::Operation(op) => op.update_matrix_world(parent),
            CsgNode::Group(g) => g.update_matrix_world(parent),
        }
    }
}

/// The operations under `node`, groups flattened away (`flatTraverse`).
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

/// `traverse` for an operation: re-evaluate it against its children when any
/// of them changed. Whether this subtree changed.
fn traverse_operation(op: &mut CsgOperation, evaluator: &mut CsgEvaluator) -> bool {
    let mut did_change = false;
    for child in &mut op.children {
        did_change = traverse_node(child, evaluator) || did_change;
    }
    let is_dirty = op.is_dirty();
    if is_dirty {
        op.mark_updated();
    }
    if !did_change {
        return is_dirty;
    }
    let mut child_ops: Vec<&CsgOperation> = Vec::new();
    for child in &op.children {
        flat_traverse(child, &mut child_ops);
    }
    // The first child against this operation's own brush; each later one
    // against the running result, a new brush at the origin.
    let mut result: Option<BufferGeometry> = None;
    for child in child_ops {
        let mut a = match result.take() {
            None => {
                let mut a = CsgBrush::new(op.brush.geometry.clone());
                a.set_matrix_world(op.brush.matrix_world);
                a
            }
            Some(geometry) => CsgBrush::new(geometry),
        };
        let mut b = CsgBrush::new(child.brush.geometry.clone());
        b.set_matrix_world(child.brush.matrix_world);
        result = Some(evaluator.evaluate(&mut a, &mut b, child.operation));
    }
    if let Some(geometry) = result {
        op.cached_geometry = geometry;
    }
    true
}

fn traverse_node(node: &mut CsgNode, evaluator: &mut CsgEvaluator) -> bool {
    match node {
        CsgNode::Operation(op) => traverse_operation(op, evaluator),
        CsgNode::Group(g) => {
            let mut did_change = false;
            for child in &mut g.children {
                did_change = traverse_node(child, evaluator) || did_change;
            }
            let is_dirty = g.is_dirty();
            if is_dirty {
                g.mark_updated();
            }
            did_change || is_dirty
        }
    }
}

/// three-bvh-csg `Evaluator.evaluateHierarchy`: update the tree's world
/// matrices, re-evaluate what changed since the last call, and return the
/// root with its children applied. Like three-bvh-csg, a root that has never
/// had a changed child to apply returns an empty geometry.
pub fn evaluate_hierarchy(root: &mut CsgOperation, evaluator: &mut CsgEvaluator) -> BufferGeometry {
    root.update_matrix_world(None);
    traverse_operation(root, evaluator);
    root.cached_geometry.clone()
}
