// The file is the test module; the inner `mod tests` keeps it consistent
// with every other test module in the crate. The CSG is checked against
// three-bvh-csg itself in `tests/csg_upstream.rs`; these are the properties
// that hold whatever the reference.
#[allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::core::BufferGeometry;
    use crate::geometries::BoxGeometry;
    use crate::{
        evaluate_hierarchy, CsgBrush, CsgEvaluator, CsgNode, CsgOperation, CsgOperationGroup,
        ADDITION, INTERSECTION, SUBTRACTION,
    };

    /// Signed volume enclosed by a triangle soup.
    fn volume(g: &BufferGeometry) -> f64 {
        let p = &g.get_attribute("position").unwrap().array;
        p.chunks_exact(9)
            .map(|t| {
                let (a, b, c) = (
                    [t[0] as f64, t[1] as f64, t[2] as f64],
                    [t[3] as f64, t[4] as f64, t[5] as f64],
                    [t[6] as f64, t[7] as f64, t[8] as f64],
                );
                let bxc = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ];
                (a[0] * bxc[0] + a[1] * bxc[1] + a[2] * bxc[2]) / 6.0
            })
            .sum()
    }

    #[test]
    fn subtracting_a_nested_box_leaves_its_faces_as_a_cavity() {
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let mut outer = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut inner = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        let result = evaluator.evaluate(&mut outer, &mut inner, SUBTRACTION);
        // Both boxes whole: twelve triangles each, the inner ones turned in.
        assert_eq!(result.get_attribute("position").unwrap().count(), 72);
        let want = 4.0 * 2.5 * 2.5 - 3.6 * 2.1 * 2.1;
        assert!(
            (volume(&result) - want).abs() < 1e-4,
            "volume {}",
            volume(&result)
        );
    }

    /// three-bvh-csg's defaults: `position`, `uv` and `normal` carried, and a
    /// group per source group — `a`'s six box faces, then `b`'s from six on.
    #[test]
    fn the_result_keeps_attributes_and_source_groups() {
        let mut evaluator = CsgEvaluator::new();
        let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
        let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
        b.set_position(1.0, 0.0, 0.0);
        b.update_matrix_world(None);
        let result = evaluator.evaluate(&mut a, &mut b, ADDITION);
        for name in ["position", "uv", "normal"] {
            assert!(result.get_attribute(name).is_some(), "no {name}");
        }
        let n = result.get_attribute("position").unwrap().count() as u32;
        assert_eq!(
            result.groups.iter().map(|g| g.count).sum::<u32>(),
            n,
            "groups cover every vertex"
        );
        assert!(
            result.groups.iter().any(|g| g.material_index < 6),
            "a's groups"
        );
        assert!(
            result.groups.iter().any(|g| g.material_index >= 6),
            "b's groups"
        );
    }

    /// Operations evaluated together give what each does alone, and the
    /// part of A outside B and the part inside it make up A.
    #[test]
    fn several_operations_in_one_pass_partition_the_solid() {
        let brushes = || {
            let a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
            let mut b = CsgBrush::new(BoxGeometry::new(1.5, 1.5, 1.5));
            b.set_position(0.9, 0.4, 0.3);
            b.update_matrix_world(None);
            (a, b)
        };
        let mut evaluator = CsgEvaluator::new();
        let (mut a, mut b) = brushes();
        let both = evaluator.evaluate_ops(&mut a, &mut b, &[SUBTRACTION, INTERSECTION]);
        let (mut a, mut b) = brushes();
        let alone = evaluator.evaluate(&mut a, &mut b, SUBTRACTION);
        assert_eq!(
            both[0].get_attribute("position").unwrap().array,
            alone.get_attribute("position").unwrap().array
        );
        let total = volume(&both[0]) + volume(&both[1]);
        assert!(
            (total - 8.0).abs() < 1e-4,
            "A − B and A ∩ B enclose {total}, not A's 8"
        );
    }

    /// A group's transform reaches the operations inside it: a box cut by a
    /// child of a group at x = 1 is the box cut by the same child placed
    /// there directly — half the unit cut lies inside.
    #[test]
    fn a_group_transform_composes_into_its_children() {
        let evaluate = |grouped: bool| {
            let mut root = CsgOperation::new(BoxGeometry::new(2.0, 2.0, 2.0), ADDITION);
            let mut cut = CsgOperation::new(BoxGeometry::new(1.0, 1.0, 1.0), SUBTRACTION);
            if grouped {
                let mut group = CsgOperationGroup::new();
                group.set_position(1.0, 0.0, 0.0);
                group.add_child(CsgNode::Operation(cut));
                root.add_child(CsgNode::Group(group));
            } else {
                cut.set_position(1.0, 0.0, 0.0);
                root.add_child(CsgNode::Operation(cut));
            }
            evaluate_hierarchy(&mut root, &mut CsgEvaluator::new())
        };
        let (grouped, direct) = (evaluate(true), evaluate(false));
        assert!(
            (volume(&grouped) - (8.0 - 0.5)).abs() < 1e-4,
            "volume {}",
            volume(&grouped)
        );
        assert_eq!(
            grouped.get_attribute("position").unwrap().array,
            direct.get_attribute("position").unwrap().array
        );
    }
}
