//! three-bvh-csg 0.0.16's `Evaluator`.

use crate::core::{BufferAttribute, BufferGeometry, GeometryGroup};

use super::attribute_data::TypedAttributeData;
use super::brush::CsgBrush;
use super::constants::Operation;
use super::operations::perform_operation;
use super::triangle_splitter::TriangleSplitter;

/// three-bvh-csg's `Evaluator`: booleans between [`CsgBrush`]es, carrying
/// the vertex [`attributes`](Self::attributes) named and, with
/// [`use_groups`](Self::use_groups), which source group each triangle came
/// from.
#[derive(Debug)]
pub struct CsgEvaluator {
    /// Keep `a`'s groups and then `b`'s as separate groups of the result
    /// (three-bvh-csg's default, `true`); `false` makes one.
    pub use_groups: bool,
    /// The vertex attributes the result carries, interpolated across each
    /// cut — three-bvh-csg's default `position`, `uv`, `normal`. Ones `a`
    /// does not have are skipped (three-bvh-csg throws).
    pub attributes: Vec<String>,
    splitter: TriangleSplitter,
}

impl Default for CsgEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// One group of one operation's result: each attribute's name, item size
/// and data.
pub type GroupArrays = Vec<(String, usize, Vec<f32>)>;

impl CsgEvaluator {
    pub fn new() -> Self {
        Self {
            use_groups: true,
            attributes: ["position", "uv", "normal"].map(String::from).to_vec(),
            splitter: TriangleSplitter::new(),
        }
    }

    /// `a` against `b` under `operation`: a non-indexed triangle soup in world
    /// space, grouped by source group when [`use_groups`](Self::use_groups).
    pub fn evaluate(
        &mut self,
        a: &mut CsgBrush,
        b: &mut CsgBrush,
        operation: Operation,
    ) -> BufferGeometry {
        self.evaluate_ops(a, b, &[operation])
            .pop()
            .unwrap_or_default()
    }

    /// Several operations in one pass over the same split triangles (three-bvh-csg's
    /// `evaluate(a, b, [ops], [targets])`): one result per operation.
    pub fn evaluate_ops(
        &mut self,
        a: &mut CsgBrush,
        b: &mut CsgBrush,
        operations: &[Operation],
    ) -> Vec<BufferGeometry> {
        let names: Vec<String> = self
            .attributes
            .iter()
            .filter(|n| a.geometry.get_attribute(n).is_some())
            .cloned()
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let results = self
            .evaluate_groups(a, b, operations, &names)
            .unwrap_or_default();
        results
            .into_iter()
            .map(|groups| {
                let mut geometry = BufferGeometry::new();
                let mut start = 0u32;
                for (index, group) in groups.iter().enumerate() {
                    let count = group.first().map_or(0, |(_, size, data)| data.len() / size) as u32;
                    if self.use_groups && count != 0 {
                        geometry.groups.push(GeometryGroup {
                            start,
                            count,
                            material_index: index as u32,
                        });
                    }
                    start += count;
                }
                for (k, &name) in names.iter().enumerate() {
                    let size = groups.first().map_or(0, |g| g[k].1);
                    let data: Vec<f32> =
                        groups.iter().flat_map(|g| g[k].2.iter().copied()).collect();
                    if size != 0 {
                        geometry.set_attribute(name, BufferAttribute::new(data, size));
                    }
                }
                geometry
            })
            .collect()
    }

    /// three-bvh-csg's `Evaluator.evaluate` data, before it becomes geometry
    /// ([`GroupArrays`]):
    /// for each of `operations`, each group's arrays of the `attributes`
    /// named (in that order, each with its item size), group `g` holding the
    /// triangles from `a`'s group `g` and from `b`'s group `g − offset`
    /// (`a.geometry.groups.length || 1`). Without `use_groups` there is one
    /// group. Every attribute must exist on `a`, as three-bvh-csg requires.
    pub fn evaluate_groups(
        &mut self,
        a: &mut CsgBrush,
        b: &mut CsgBrush,
        operations: &[Operation],
        attributes: &[&str],
    ) -> Result<Vec<Vec<GroupArrays>>, String> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }
        let mut layout = Vec::with_capacity(attributes.len());
        for &name in attributes {
            let attr = a
                .geometry
                .get_attribute(name)
                .ok_or_else(|| format!("Evaluator: brush geometry has no '{name}' attribute"))?;
            layout.push((name, attr.item_size));
        }
        let mut data: Vec<TypedAttributeData> = operations
            .iter()
            .map(|_| {
                let mut d = TypedAttributeData::new();
                for &(name, size) in &layout {
                    d.initialize_array(name, size);
                }
                d
            })
            .collect();
        perform_operation(
            a,
            b,
            operations,
            &mut self.splitter,
            &mut data,
            self.use_groups,
        );
        Ok(data
            .into_iter()
            .map(|mut d| {
                (0..d.group_count)
                    .map(|g| {
                        let set = &mut d.group_attributes[g];
                        layout
                            .iter()
                            .map(|&(name, size)| {
                                let arr = set.remove(name).map(|t| t.data).unwrap_or_default();
                                (name.to_string(), size, arr)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }

    pub fn reset(&mut self) {
        self.splitter.reset();
    }
}
