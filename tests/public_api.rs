//! Pieces made public for downstream code, used the way a dependent crate
//! would — through the crate's public paths only.

use threers::curves::interpolations::{catmull_rom, cubic_bezier, quadratic_bezier};

/// three.js `Interpolations`: the same values three.js r165 returns.
#[test]
fn interpolations_match_three_js() {
    assert_eq!(quadratic_bezier(0.5, 0.0, 1.0, 0.0), 0.5);
    assert_eq!(cubic_bezier(0.5, 0.0, 0.0, 1.0, 1.0), 0.5);
    assert_eq!(cubic_bezier(0.25, 0.0, 1.0, 1.0, 0.0), 0.5625);
    // Catmull-Rom passes through its inner points.
    assert_eq!(catmull_rom(0.0, 9.0, 2.0, 5.0, -7.0), 2.0);
    assert_eq!(catmull_rom(1.0, 9.0, 2.0, 5.0, -7.0), 5.0);
}

/// Walking a BVH by hand reaches every triangle exactly once, inside the
/// bounds of every node above it.
#[cfg(feature = "mesh-bvh")]
#[test]
fn a_bvh_can_be_walked_node_by_node() {
    use threers::{BoxGeometry, BvhNodeView, MeshBvh, SphereGeometry};
    for geometry in [
        SphereGeometry::new(1.0, 48, 32),
        BoxGeometry::new(1.0, 2.0, 3.0),
    ] {
        let bvh = MeshBvh::build(&geometry, Default::default()).expect("bvh");
        let (mut seen, mut stack) = (
            vec![false; bvh.triangle_count()],
            vec![(0usize, bvh.bounding_box())],
        );
        while let Some((i, parent)) = stack.pop() {
            let node = bvh.node(i).expect("in range");
            let b = node.bounds();
            assert!(
                parent.contains_box(&b) || parent == b,
                "node {i} outside its parent"
            );
            match node {
                BvhNodeView::Internal { left, right, .. } => stack.extend([(left, b), (right, b)]),
                BvhNodeView::Leaf { offset, count, .. } => {
                    for (k, slot) in seen.iter_mut().enumerate().skip(offset).take(count) {
                        assert!(!*slot, "triangle {k} reached twice");
                        *slot = true;
                        let t = bvh.triangle_at_order_index(k);
                        let mut slack = b;
                        slack.expand_by_scalar(1e-5);
                        for p in [t.a, t.b, t.c] {
                            assert!(slack.contains_point(p));
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "some triangles are in no leaf");
        assert!(bvh.node(bvh.node_count()).is_none());
    }
}

/// A closed solid is watertight; a sheet, or a solid with a face missing, is not.
#[cfg(feature = "openscad")]
#[test]
fn watertightness_is_checkable() {
    use threers::exact_csg::is_watertight;
    use threers::{BoxGeometry, PlaneGeometry, SphereGeometry};
    assert!(is_watertight(&BoxGeometry::new(1.0, 1.0, 1.0)));
    assert!(is_watertight(&SphereGeometry::new(1.0, 24, 16)));
    assert!(!is_watertight(&PlaneGeometry::new(1.0, 1.0)));
    let mut open = BoxGeometry::new(1.0, 1.0, 1.0);
    let index = open.index.take().unwrap();
    open.set_index(index[6..].to_vec());
    assert!(!is_watertight(&open));
}
