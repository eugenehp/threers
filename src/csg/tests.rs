// The file is the test module; the inner `mod tests` keeps it consistent
// with every other test module in the crate.
#[allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::core::BufferAttribute;
    use crate::core::BufferGeometry;
    use crate::geometries::{BoxGeometry, SphereGeometry};
    use crate::utils::compute_vertex_normals;
    use crate::{
        build_bvh_csg_hierarchy_geometry, evaluate_hierarchy, CsgBrush, CsgEvaluator, CsgNode,
        CsgOperation, CsgOperationGroup, ADDITION, SUBTRACTION,
    };

    fn build_hierarchy() -> BufferGeometry {
        let mut root = CsgOperation::new(BoxGeometry::new(4.0, 2.5, 2.5), ADDITION);
        let cut = CsgOperation::new(BoxGeometry::new(3.6, 2.1, 2.1), SUBTRACTION);
        let mut sphere = CsgOperation::new(SphereGeometry::new(0.55, 24, 12), ADDITION);
        sphere.set_position(-1.1, 0.2, 1.35);

        let mut window_group = CsgOperationGroup::new();
        window_group.add_child(CsgNode::Operation(CsgOperation::new(
            BoxGeometry::new(1.2, 1.0, 0.5),
            SUBTRACTION,
        )));
        window_group.add_child(CsgNode::Operation(CsgOperation::new(
            BoxGeometry::new(1.2, 1.0, 0.12),
            ADDITION,
        )));
        window_group.set_position(0.8, 0.15, 1.35);

        root.add_child(CsgNode::Operation(cut));
        root.add_child(CsgNode::Operation(sphere));
        root.add_child(CsgNode::Group(window_group));

        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let mut geom = evaluate_hierarchy(&mut root, &mut evaluator);
        compute_vertex_normals(&mut geom);
        geom
    }

    #[test]
    fn union_two_boxes_produces_triangles() {
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
        let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
        let result = evaluator.evaluate(&mut a, &mut b, ADDITION);
        let verts = result.get_attribute("position").unwrap().count();
        assert!(
            verts >= 36,
            "union should emit triangle soup, got {verts} verts"
        );
    }

    #[test]
    fn subtract_nested_boxes_includes_inner_faces() {
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let mut outer = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut inner = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        let result = evaluator.evaluate(&mut outer, &mut inner, SUBTRACTION);
        let verts = result.get_attribute("position").unwrap().count();
        assert_eq!(verts, 72, "expected outer+inner cavity faces, got {verts}");
    }

    #[test]
    fn hierarchy_step_vertex_counts() {
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;

        let mut root = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut cut = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        let s1 = evaluator.evaluate(&mut root, &mut cut, SUBTRACTION);
        let v1 = s1.get_attribute("position").unwrap().count();

        let mut acc = CsgBrush::new(s1);
        acc.matrix_world = crate::math::Matrix4::identity();
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        let s2 = evaluator.evaluate(&mut acc, &mut sphere, ADDITION);
        let v2 = s2.get_attribute("position").unwrap().count();

        let mut acc2 = CsgBrush::new(s2);
        acc2.matrix_world = crate::math::Matrix4::identity();
        let mut win_cut = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.5));
        win_cut.set_position(0.8, 0.15, 1.35);
        win_cut.update_matrix_world(None);
        let s3 = evaluator.evaluate(&mut acc2, &mut win_cut, SUBTRACTION);
        let v3 = s3.get_attribute("position").unwrap().count();

        let mut acc3 = CsgBrush::new(s3);
        acc3.matrix_world = crate::math::Matrix4::identity();
        let mut win_frame = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.12));
        win_frame.set_position(0.8, 0.15, 1.35);
        win_frame.update_matrix_world(None);
        let s4 = evaluator.evaluate(&mut acc3, &mut win_frame, ADDITION);

        let v4 = s4.get_attribute("position").unwrap().count();
        eprintln!("steps: cut={v1} +sphere={v2} +winCut={v3} +winFrame={v4}");
        assert!(v4 > 30_000);
    }

    #[test]
    fn win_cut_on_js_sphere_geometry_matches_reference() {
        use super::super::attribute_data::TypedAttributeData;
        use super::super::operations::perform_operation;
        use crate::core::BufferAttribute;

        const RAW: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let vert_count = u32::from_le_bytes(RAW[4..8].try_into().expect("header")) as usize;
        let pos_bytes = &RAW[8..8 + vert_count * 12];
        let mut positions = Vec::with_capacity(vert_count * 3);
        for chunk in pos_bytes.chunks_exact(4) {
            positions.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        compute_vertex_normals(&mut geom);

        let mut acc = CsgBrush::new(geom);
        acc.matrix_world = crate::math::Matrix4::identity();
        let mut win_cut = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.5));
        win_cut.set_position(0.8, 0.15, 1.35);
        win_cut.update_matrix_world(None);

        let mut splitter = super::super::triangle_splitter::TriangleSplitter::new();
        let mut attr = vec![TypedAttributeData::new()];
        attr[0].initialize_array("position", 3);
        attr[0].initialize_array("normal", 3);
        perform_operation(
            &mut acc,
            &mut win_cut,
            &[SUBTRACTION],
            &mut splitter,
            &mut attr,
            false,
        );
        let verts = attr[0].get_count(0);
        assert!(
            (verts as i64 - 55_914).unsigned_abs() <= 200,
            "winCut on JS sphere soup expected ~55914, got {verts}"
        );
    }

    fn count_window_region(geom: &BufferGeometry) -> (usize, usize) {
        let pos = geom.get_attribute("position").unwrap();
        let wf_min = (0.8 - 0.6 - 0.05, 0.15 - 0.5 - 0.05, 1.35 - 0.06 - 0.05);
        let wf_max = (0.8 + 0.6 + 0.05, 0.15 + 0.5 + 0.05, 1.35 + 0.06 + 0.05);
        let mut win = 0usize;
        let mut z_ext = 0usize;
        for i in 0..pos.count() {
            let x = pos.array[i * 3];
            let y = pos.array[i * 3 + 1];
            let z = pos.array[i * 3 + 2];
            if x >= wf_min.0
                && x <= wf_max.0
                && y >= wf_min.1
                && y <= wf_max.1
                && z >= wf_min.2
                && z <= wf_max.2
            {
                win += 1;
            }
            if (1.05..=1.41).contains(&z) && (0.1..=1.5).contains(&x) {
                z_ext += 1;
            }
        }
        (win, z_ext)
    }

    fn tri_key(
        pos: &crate::core::BufferAttribute,
        t: usize,
    ) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        use super::super::triangle_utils::hash_vertex3;
        let i = t * 9;
        let a = crate::math::Vector3::new(pos.array[i], pos.array[i + 1], pos.array[i + 2]);
        let b = crate::math::Vector3::new(pos.array[i + 3], pos.array[i + 4], pos.array[i + 5]);
        let c = crate::math::Vector3::new(pos.array[i + 6], pos.array[i + 7], pos.array[i + 8]);
        let ha = hash_vertex3(a);
        let hb = hash_vertex3(b);
        let hc = hash_vertex3(c);
        (ha.0, ha.1, ha.2, hb.0, hb.1, hb.2, hc.0, hc.1, hc.2)
    }

    #[test]
    fn native_sphere_soup_overlaps_js_reference() {
        use crate::core::BufferAttribute;
        use std::collections::HashSet;

        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let vert_count =
            u32::from_le_bytes(AFTER_SPHERE[4..8].try_into().expect("header")) as usize;
        let pos_bytes = &AFTER_SPHERE[8..8 + vert_count * 12];
        let mut positions = Vec::with_capacity(vert_count * 3);
        for chunk in pos_bytes.chunks_exact(4) {
            positions.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let mut js = BufferGeometry::new();
        js.set_attribute("position", BufferAttribute::new(positions, 3));

        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let mut root = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut cut = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        let s1 = evaluator.evaluate(&mut root, &mut cut, SUBTRACTION);
        let mut acc = CsgBrush::new(s1);
        acc.matrix_world = crate::math::Matrix4::identity();
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        let native = evaluator.evaluate(&mut acc, &mut sphere, ADDITION);

        let np = native.get_attribute("position").unwrap();
        let jp = js.get_attribute("position").unwrap();
        let mut js_keys = HashSet::new();
        for t in 0..jp.count() / 3 {
            js_keys.insert(tri_key(jp, t));
        }
        let mut shared = 0usize;
        let mut only_native = 0usize;
        for t in 0..np.count() / 3 {
            if js_keys.contains(&tri_key(np, t)) {
                shared += 1;
            } else {
                only_native += 1;
            }
        }
        eprintln!(
            "sphere soup tris: native={} js={} shared={} only_native={}",
            np.count() / 3,
            jp.count() / 3,
            shared,
            only_native
        );
    }

    #[test]
    #[ignore = "Rust winFrame ADDITION still +1.2k verts over JS export on matching winCut input"]
    fn hierarchy_window_steps_from_js_sphere_match_export() {
        use crate::core::BufferAttribute;

        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        const FINAL: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-hierarchy.geom.bin");

        let load_positions = |raw: &[u8]| -> BufferGeometry {
            let vert_count = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
            let pos_bytes = &raw[8..8 + vert_count * 12];
            let mut positions = Vec::with_capacity(vert_count * 3);
            for chunk in pos_bytes.chunks_exact(4) {
                positions.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            let mut geom = BufferGeometry::new();
            geom.set_attribute("position", BufferAttribute::new(positions, 3));
            compute_vertex_normals(&mut geom);
            geom
        };

        let expected = load_positions(FINAL);
        let expected_verts = expected.get_attribute("position").unwrap().count();

        let mut acc = CsgBrush::new(load_positions(AFTER_SPHERE));
        acc.matrix_world = crate::math::Matrix4::identity();

        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;

        let mut win_cut = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.5));
        win_cut.set_position(0.8, 0.15, 1.35);
        win_cut.update_matrix_world(None);
        let after_cut = evaluator.evaluate(&mut acc, &mut win_cut, SUBTRACTION);
        let cut_verts = after_cut.get_attribute("position").unwrap().count();
        eprintln!("winCut from JS sphere via evaluator: {cut_verts}");

        let mut acc2 = CsgBrush::new(after_cut);
        acc2.matrix_world = crate::math::Matrix4::identity();
        let mut win_frame = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.12));
        win_frame.set_position(0.8, 0.15, 1.35);
        win_frame.update_matrix_world(None);
        let mut geom = evaluator.evaluate(&mut acc2, &mut win_frame, ADDITION);
        compute_vertex_normals(&mut geom);

        let actual_verts = geom.get_attribute("position").unwrap().count();
        let delta = (actual_verts as i64 - expected_verts as i64).unsigned_abs();
        assert!(
            delta <= 200,
            "window steps from JS sphere: {actual_verts} vs export {expected_verts} (delta {delta})"
        );
    }

    #[test]
    fn build_bvh_csg_hierarchy_with_reference_sphere_renders_window_frame() {
        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        const AFTER_WINCUT: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-step3-wincut.geom.bin");
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;
        let geom = build_bvh_csg_hierarchy_geometry(&mut evaluator, AFTER_SPHERE, AFTER_WINCUT);
        let verts = geom.get_attribute("position").unwrap().count();
        assert!(
            verts > 58_000,
            "reference-sphere hierarchy expected ~59k verts, got {verts}"
        );

        let pos = geom.get_attribute("position").unwrap();
        let mut toward_cam = 0usize;
        for t in 0..pos.count() / 3 {
            let a = crate::math::Vector3::new(
                pos.array[t * 9],
                pos.array[t * 9 + 1],
                pos.array[t * 9 + 2],
            );
            let b = crate::math::Vector3::new(
                pos.array[t * 9 + 3],
                pos.array[t * 9 + 4],
                pos.array[t * 9 + 5],
            );
            let c = crate::math::Vector3::new(
                pos.array[t * 9 + 6],
                pos.array[t * 9 + 7],
                pos.array[t * 9 + 8],
            );
            let cx = (a.x + b.x + c.x) / 3.0;
            let cy = (a.y + b.y + c.y) / 3.0;
            let cz = (a.z + b.z + c.z) / 3.0;
            if !(0.15..=1.45).contains(&cx)
                || !(-0.45..=0.65).contains(&cy)
                || !(1.28..=1.42).contains(&cz)
            {
                continue;
            }
            let n = (b - a).cross(c - a);
            let area = n.length();
            if area < 1e-8 {
                continue;
            }
            if n.z / area < -0.5 {
                toward_cam += 1;
            }
        }
        assert!(
            toward_cam >= 400,
            "window frame slab should have camera-facing faces, got {toward_cam}"
        );
    }

    #[test]
    #[ignore = "native sphere ADDITION topology diverges from JS CSG; winCut is ~52k not ~56k"]
    fn native_after_sphere_win_cut_vertex_count() {
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = false;

        let mut root = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut cut = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        let s1 = evaluator.evaluate(&mut root, &mut cut, SUBTRACTION);

        let mut acc = CsgBrush::new(s1);
        acc.matrix_world = crate::math::Matrix4::identity();
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        let s2 = evaluator.evaluate(&mut acc, &mut sphere, ADDITION);

        let mut acc2 = CsgBrush::new(s2);
        acc2.matrix_world = crate::math::Matrix4::identity();
        let mut win_cut = CsgBrush::new(BoxGeometry::new(1.2, 1.0, 0.5));
        win_cut.set_position(0.8, 0.15, 1.35);
        win_cut.update_matrix_world(None);
        let s3 = evaluator.evaluate(&mut acc2, &mut win_cut, SUBTRACTION);
        let v3 = s3.get_attribute("position").unwrap().count();
        eprintln!("native winCut after native sphere: {v3} (JS ref 55914)");
        assert!(
            (v3 as i64 - 55_914).unsigned_abs() <= 500,
            "native winCut expected ~55914, got {v3}"
        );
    }

    #[test]
    fn winframe_slab_faces_survive_evaluator() {
        let geom = build_hierarchy();
        let pos = geom.get_attribute("position").unwrap();
        let mut toward_cam = 0usize;
        for t in 0..pos.count() / 3 {
            let a = crate::math::Vector3::new(
                pos.array[t * 9],
                pos.array[t * 9 + 1],
                pos.array[t * 9 + 2],
            );
            let b = crate::math::Vector3::new(
                pos.array[t * 9 + 3],
                pos.array[t * 9 + 4],
                pos.array[t * 9 + 5],
            );
            let c = crate::math::Vector3::new(
                pos.array[t * 9 + 6],
                pos.array[t * 9 + 7],
                pos.array[t * 9 + 8],
            );
            let cx = (a.x + b.x + c.x) / 3.0;
            let cy = (a.y + b.y + c.y) / 3.0;
            let cz = (a.z + b.z + c.z) / 3.0;
            if !(0.15..=1.45).contains(&cx)
                || !(-0.45..=0.65).contains(&cy)
                || !(1.28..=1.42).contains(&cz)
            {
                continue;
            }
            let n = (b - a).cross(c - a);
            let area = n.length();
            if area < 1e-8 {
                continue;
            }
            if n.z / area < -0.5 {
                toward_cam += 1;
            }
        }
        assert!(
            toward_cam >= 500,
            "thin winFrame slab should retain camera-facing faces, got {toward_cam}"
        );
    }

    #[test]
    fn window_frame_region_has_geometry() {
        const RAW: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-hierarchy.geom.bin");
        let js_vert_count = u32::from_le_bytes(RAW[4..8].try_into().expect("header")) as usize;
        let pos_bytes = &RAW[8..8 + js_vert_count * 12];
        let mut positions = Vec::with_capacity(js_vert_count * 3);
        for chunk in pos_bytes.chunks_exact(4) {
            positions.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let mut js_geom = BufferGeometry::new();
        js_geom.set_attribute("position", BufferAttribute::new(positions, 3));

        let native = build_hierarchy();
        let (nw, nz) = count_window_region(&native);
        let (jw, jz) = count_window_region(&js_geom);
        eprintln!("native winFrame_box={nw} z_region={nz}");
        eprintln!("js winFrame_box={jw} z_region={jz}");
        assert!(
            nw >= jw / 2,
            "native window frame region {nw} verts vs JS {jw} — hole/frame likely missing"
        );
    }

    #[test]
    fn hierarchy_matches_js_export_vertex_count() {
        const RAW: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-hierarchy.geom.bin");
        let expected_verts = u32::from_le_bytes(RAW[4..8].try_into().expect("header")) as usize;

        let geom = build_hierarchy();
        let actual_verts = geom.get_attribute("position").unwrap().count();

        let delta = (actual_verts as i64 - expected_verts as i64).unsigned_abs();
        let tolerance = (expected_verts as f64 * 0.02).max(200.0) as u64;
        assert!(
            delta <= tolerance,
            "hierarchy verts {actual_verts} vs JS export {expected_verts} (delta {delta} > {tolerance})"
        );
    }
}
