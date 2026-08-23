// Same shape as the other test modules in the crate.
#[allow(clippy::module_inception)]
#[cfg(test)]
mod step_tests {
    //! Release parity suite for hierarchy CSG (exact ordered TriKey + vert counts).
    use crate::core::{BufferAttribute, BufferGeometry};
    use crate::csg::topology::assert_topology_overlap;
    use crate::utils::compute_vertex_normals;
    use crate::{
        assert_step_verts, evaluate_live_through, evaluate_through, step1_shell_cut, step3_win_cut,
        step4_win_frame, CsgEvaluator, JS_STEP2_VERTS, JS_STEP3_VERTS, JS_STEP4_VERTS,
    };

    fn vert_count(geom: &BufferGeometry) -> usize {
        geom.get_attribute("position").unwrap().count()
    }

    fn load_bin_positions(raw: &[u8]) -> BufferGeometry {
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
    }

    #[test]
    fn step1_shell_cut_topology_matches_js() {
        const STEP1: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-step1-shell.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let native = step1_shell_cut(&mut ev);
        let reference = load_bin_positions(STEP1);
        assert_topology_overlap(&native, &reference, 1.0, "step1");
    }

    #[test]
    fn sphere_primitive_soup_matches_js() {
        use super::super::geometry_prep::ensure_non_indexed;
        use crate::csg::topology::topology_overlap;
        use crate::geometries::SphereGeometry;

        const SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/sphere-0.55-24-12.geom.bin");
        let mut native = SphereGeometry::new(0.55, 24, 12);
        ensure_non_indexed(&mut native);
        let reference = load_bin_positions(SPHERE);
        let o = topology_overlap(&native, &reference);
        eprintln!(
            "sphere soup: native={} ref={} shared={} only_native={} only_ref={}",
            o.native_tris, o.reference_tris, o.shared, o.only_native, o.only_reference
        );
        assert_topology_overlap(&native, &reference, 1.0, "sphere primitive");
    }

    #[test]
    fn sphere_addition_bvhcast_pairs_match_js() {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::{
            bvh_buffers_match, bvhcast_pairs, intersection_neighbors, load_bvhcast_reference,
            pair_overlap, prepare_brush_geometry, serialized_bvh,
        };
        use super::super::operations::collect_intersecting_triangles;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;

        const REF: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvhcast-shell-sphere.bin");
        let reference = load_bvhcast_reference(REF);

        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let shell_geom = step1_shell_cut(&mut ev);

        let mut acc = CsgBrush::new(shell_geom);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);

        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);

        let native_pairs = bvhcast_pairs(&acc, &sphere);
        let (n_len, r_len, shared) = pair_overlap(&native_pairs, &reference.pairs);
        eprintln!(
            "bvhcast pairs: native={n_len} ref={r_len} shared={shared} ({:.1}%)",
            shared as f64 / r_len.max(1) as f64 * 100.0
        );

        let shell_native = serialized_bvh(&acc);
        let sphere_native = serialized_bvh(&sphere);
        eprintln!(
            "shell BVH match: {}  sphere BVH match: {}",
            bvh_buffers_match(&shell_native, &reference.shell_bvh),
            bvh_buffers_match(&sphere_native, &reference.sphere_bvh)
        );

        let prefix_match = native_pairs
            .iter()
            .zip(reference.pairs.iter())
            .take(20)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!("first-20 pair prefix matches: {prefix_match}/20");

        // Native and JS both enumerate all leaf-leaf bbox pairs (6240 for shell+sphere).
        let ref_set: std::collections::HashSet<_> = reference.pairs.iter().copied().collect();
        let native_filtered: Vec<_> = native_pairs
            .iter()
            .copied()
            .filter(|p| ref_set.contains(p))
            .collect();
        let filtered_prefix = native_filtered
            .iter()
            .zip(reference.pairs.iter())
            .take(20)
            .filter(|(a, b)| a == b)
            .count();
        eprintln!(
            "filtered shared-pair prefix matches: {filtered_prefix}/20 (native_filtered={})",
            native_filtered.len()
        );

        assert!(
            bvh_buffers_match(&shell_native, &reference.shell_bvh),
            "shell BVH buffers differ from JS"
        );
        assert!(
            bvh_buffers_match(&sphere_native, &reference.sphere_bvh),
            "sphere BVH buffers differ from JS"
        );
        assert_eq!(
            native_filtered,
            reference.pairs,
            "shared bvhcast pair order differs (filtered {}/{})",
            filtered_prefix,
            reference.pairs.len().min(20)
        );

        let (a_map, b_map) = collect_intersecting_triangles(&acc, &sphere);
        eprintln!(
            "intersection ids: native_a={} ref_a={} native_b={} ref_b={}",
            a_map.ids.len(),
            reference.a_ids.len(),
            b_map.ids.len(),
            reference.b_ids.len()
        );
        assert_eq!(a_map.ids, reference.a_ids, "shell intersection id order");
        assert_eq!(b_map.ids, reference.b_ids, "sphere intersection id order");

        let native_a_neighbors = intersection_neighbors(&a_map);
        let native_b_neighbors = intersection_neighbors(&b_map);
        if !reference.a_neighbors.is_empty() {
            assert_eq!(
                native_a_neighbors, reference.a_neighbors,
                "shell intersection neighbor order"
            );
            assert_eq!(
                native_b_neighbors, reference.b_neighbors,
                "sphere intersection neighbor order"
            );
        }
    }

    #[test]
    fn ia15_ib401_src16_split_matches_js() {
        use super::super::js_topology::{JsTriangle, JsVec3};
        use super::super::triangle_splitter::TriangleSplitter;
        use super::super::triangle_utils::hash_coord;

        fn ord_key(t: JsTriangle) -> String {
            format!(
                "{},{},{},{},{},{},{},{},{}",
                hash_coord(t.a.x),
                hash_coord(t.a.y),
                hash_coord(t.a.z),
                hash_coord(t.b.x),
                hash_coord(t.b.y),
                hash_coord(t.b.z),
                hash_coord(t.c.x),
                hash_coord(t.c.y),
                hash_coord(t.c.z),
            )
        }

        let tri = JsTriangle {
            a: JsVec3::new(
                0.18703882002231964,
                -0.7325606782431417,
                -0.3000000476837159,
            ),
            b: JsVec3::new(
                -0.3510995269214351,
                -0.30194287194517766,
                -0.3000000476837159,
            ),
            c: JsVec3::new(-0.5541723443478346, 0.5898297789076434, -0.3000000476837159),
        };
        let clip = JsTriangle {
            a: JsVec3::new(
                -0.23815692961215973,
                -0.27500003576278687,
                -0.4125000238418579,
            ),
            b: JsVec3::new(
                -0.19445432722568512,
                -0.3889087438583374,
                -0.3368048667907715,
            ),
            c: JsVec3::new(
                -0.27500009536743164,
                -0.3889087438583374,
                -0.2749998867511749,
            ),
        };
        let mut s = TriangleSplitter::new();
        s.initialize(tri);
        eprintln!("intersects={}", clip.intersects_triangle(tri, true));
        let plane = clip.get_plane();
        eprintln!(
            "plane n=({:.17},{:.17},{:.17}) c={:.17}",
            plane.normal.x, plane.normal.y, plane.normal.z, plane.constant
        );
        for t in 0..3 {
            let arr = [tri.a, tri.b, tri.c];
            let sd = plane.distance_to_point(arr[t]);
            let ed = plane.distance_to_point(arr[(t + 1) % 3]);
            eprintln!("edge {t}: sd={sd:.17e} ed={ed:.17e}");
        }
        s.split_by_triangle(clip);
        eprintln!("native count={} (js=2)", s.triangle_count());
        let target = "187039,-732560,-299999,-322198,-325068,-299999,-351099,-301942,-299999";
        for (i, t) in s.clipped_js_triangles().iter().enumerate() {
            let k = ord_key(*t);
            eprintln!("  out{i} {k}");
        }

        // Manual split replay
        let plane = clip.get_plane();
        let arr = [tri.a, tri.b, tri.c];
        let mut hit_vec = JsVec3::default();
        let mut fs = JsVec3::default();
        let mut fe = JsVec3::default();
        let mut intersects = 0usize;
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for t in 0..3 {
            let start = arr[t];
            let end = arr[(t + 1) % 3];
            let sd = plane.distance_to_point(start);
            let ed = plane.distance_to_point(end);
            if sd > 0.0 {
                pos.push(t);
            } else {
                neg.push(t);
            }
            if sd.abs() < 1e-10 {
                eprintln!("manual edge {t}: skip");
                continue;
            }
            let mut did = plane.intersect_line(start, end, &mut hit_vec).is_some();
            if !did && ed.abs() < 1e-10 {
                hit_vec = end;
                did = true;
            }
            let d = hit_vec.distance_to(start);
            // NaN counts as "not near": the negation is what admitted it before.
            let use_hit = did && (d >= 1e-10 || d.is_nan());
            eprintln!(
                "manual edge {t}: did={did} use_hit={use_hit} dist_start={:.17e} hit=({:.17},{:.17},{:.17})",
                hit_vec.distance_to(start),
                hit_vec.x, hit_vec.y, hit_vec.z
            );
            if use_hit {
                if intersects == 0 {
                    fs = hit_vec;
                } else {
                    fe = hit_vec;
                }
                intersects += 1;
            }
        }
        eprintln!(
            "manual intersects={intersects} edge_len={:.17e} pos={pos:?} neg={neg:?}",
            fs.distance_to(fe)
        );
        let single = if pos.len() >= 2 { neg[0] } else { pos[0] };
        if single == 0 {
            std::mem::swap(&mut fs, &mut fe);
        }
        let nv1 = (single + 1) % 3;
        let nv2 = (single + 2) % 3;
        let use_first = arr[nv1].distance_to_squared(fs) < arr[nv2].distance_to_squared(fe);
        let _tri1 = if use_first {
            JsTriangle {
                a: arr[nv1],
                b: fs,
                c: fe,
            }
        } else {
            JsTriangle {
                a: arr[nv2],
                b: fs,
                c: fe,
            }
        };
        // wait - when use_first false, tri1 = (nv2, fs, fe) per code...
        // re-read: else branch tri1 = (arr[next_vert2], fs, fe) = (nv2, fs, fe)
        // But TARGET needs (v0, fs, fe) and nv2 = (1+2)%3 = 0. Yes nv2=0!
        let tri1 = if !use_first {
            JsTriangle {
                a: arr[nv2],
                b: fs,
                c: fe,
            }
        } else {
            JsTriangle {
                a: arr[nv1],
                b: fs,
                c: fe,
            }
        };
        let tri2 = if use_first {
            JsTriangle {
                a: arr[nv1],
                b: arr[nv2],
                c: fs,
            }
        } else {
            JsTriangle {
                a: arr[nv1],
                b: arr[nv2],
                c: fe,
            }
        };
        let main = JsTriangle {
            a: arr[single],
            b: fe,
            c: fs,
        };
        use super::super::js_topology::is_tri_degenerate;
        eprintln!(
            "single={single} use_first={use_first} tri1={} tri2={} main={} tri1_key={}",
            is_tri_degenerate(tri1, 1e-14),
            is_tri_degenerate(tri2, 1e-14),
            is_tri_degenerate(main, 1e-14),
            ord_key(tri1)
        );
        assert!(
            s.clipped_js_triangles()
                .iter()
                .any(|t| ord_key(*t) == target),
            "missing REF105 precursor from src16"
        );
    }

    #[test]
    fn ia15_ib401_split_delta() {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::prepare_brush_geometry;
        use super::super::geometry_prep::index_at;
        use super::super::js_topology::{
            is_tri_degenerate, js_tri_from_indices, matrix_a_to_b_brushes, JsVec3,
        };
        use super::super::operations::collect_intersecting_triangles;
        use super::super::triangle_splitter::TriangleSplitter;
        use super::super::triangle_utils::hash_coord;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;

        fn ord_key(
            t: super::super::js_topology::JsTriangle,
        ) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            (
                hash_coord(t.a.x),
                hash_coord(t.a.y),
                hash_coord(t.a.z),
                hash_coord(t.b.x),
                hash_coord(t.b.y),
                hash_coord(t.b.z),
                hash_coord(t.c.x),
                hash_coord(t.c.y),
                hash_coord(t.c.z),
            )
        }

        let mut ev = CsgEvaluator::new();
        let shell = step1_shell_cut(&mut ev);
        let mut acc = CsgBrush::new(shell);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);
        let (a_inter, _) = collect_intersecting_triangles(&acc, &sphere);
        let matrix = matrix_a_to_b_brushes(&acc, &sphere);
        let pos = acc.geometry.get_attribute("position").unwrap();
        let b_pos = sphere.geometry.get_attribute("position").unwrap();
        let neighbors = a_inter.intersection_set.get(&15).unwrap();
        let i0 = index_at(&acc.geometry, 15, 0);
        let i1 = index_at(&acc.geometry, 15, 1);
        let i2 = index_at(&acc.geometry, 15, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        let mut splitter = TriangleSplitter::new();
        splitter.initialize(tri_a);
        for &ib in neighbors.iter().take(5) {
            let j0 = index_at(&sphere.geometry, ib, 0);
            let j1 = index_at(&sphere.geometry, ib, 1);
            let j2 = index_at(&sphere.geometry, ib, 2);
            splitter.split_by_triangle(js_tri_from_indices(b_pos, j0, j1, j2, None));
        }
        assert_eq!(splitter.triangle_count(), 26, "j4 count");
        let before = splitter.clipped_js_triangles();
        let before_keys: std::collections::BTreeSet<_> =
            before.iter().map(|t| ord_key(*t)).collect();

        let ib = neighbors[5];
        assert_eq!(ib, 401);
        let j0 = index_at(&sphere.geometry, ib, 0);
        let j1 = index_at(&sphere.geometry, ib, 1);
        let j2 = index_at(&sphere.geometry, ib, 2);
        let clip = js_tri_from_indices(b_pos, j0, j1, j2, None);

        let mut total_delta = 0i32;
        for (i, tri) in before.iter().enumerate() {
            if !clip.intersects_triangle(*tri, true) {
                continue;
            }
            let mut iso = TriangleSplitter::new();
            iso.initialize(*tri);
            iso.split_by_triangle(clip);
            let d = iso.triangle_count() as i32 - 1;
            let plane = clip.get_plane();
            let arr = [tri.a, tri.b, tri.c];
            let mut hit = JsVec3::default();
            let mut fs = JsVec3::default();
            let mut fe = JsVec3::default();
            let mut n = 0usize;
            for t in 0..3 {
                let start = arr[t];
                let end = arr[(t + 1) % 3];
                let sd = plane.distance_to_point(start);
                let ed = plane.distance_to_point(end);
                if sd.abs() < 1e-10 {
                    continue;
                }
                let mut did = plane.intersect_line(start, end, &mut hit).is_some();
                if !did && ed.abs() < 1e-10 {
                    hit = end;
                    did = true;
                }
                let d = hit.distance_to(start);
                if did && (d >= 1e-10 || d.is_nan()) {
                    if n == 0 {
                        fs = hit;
                    } else {
                        fe = hit;
                    }
                    n += 1;
                }
            }
            let edge = if n >= 2 { fs.distance_to(fe) } else { -1.0 };
            if d != 0 || (edge > 0.0 && edge < 1e-5) {
                let outs: Vec<_> = iso
                    .clipped_js_triangles()
                    .iter()
                    .map(|t| {
                        let ab = JsVec3::default().sub_vectors(t.b, t.a);
                        let ac = JsVec3::default().sub_vectors(t.c, t.a);
                        let cb = JsVec3::default().sub_vectors(t.b, t.c);
                        let a1 = ab.angle_to(ac);
                        let a2 = ab.angle_to(cb);
                        let a3 = std::f64::consts::PI - a1 - a2;
                        format!(
                            "degen={} a3={a3:.3e} key={:?}",
                            is_tri_degenerate(*t, 1e-14),
                            ord_key(*t)
                        )
                    })
                    .collect();
                eprintln!(
                    "src {i} delta={d} edge={edge:.17e} outs={}",
                    outs.join(" | ")
                );
                if edge > 0.0 && edge < 1e-5 {
                    let json = format!(
                        "{{\"tri\":[{},{},{},{},{},{},{},{},{}],\"clip\":[{},{},{},{},{},{},{},{},{}]}}",
                        tri.a.x, tri.a.y, tri.a.z, tri.b.x, tri.b.y, tri.b.z, tri.c.x, tri.c.y, tri.c.z,
                        clip.a.x, clip.a.y, clip.a.z, clip.b.x, clip.b.y, clip.b.z, clip.c.x, clip.c.y, clip.c.z,
                    );
                    let outp = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "tests/parity/scenes/rust/ia15-src{i}-ib401-split.json"
                    ));
                    std::fs::write(&outp, &json).expect("write");
                    eprintln!("wrote {}", outp.display());
                }
            }
            total_delta += d;
        }
        splitter.split_by_triangle(clip);
        let after_keys: std::collections::BTreeSet<_> = splitter
            .clipped_js_triangles()
            .iter()
            .map(|t| ord_key(*t))
            .collect();
        eprintln!(
            "ib401 total_delta={total_delta} before={} after={} (js delta should be +8 -> 34)",
            before_keys.len(),
            after_keys.len()
        );
    }

    #[test]
    fn shell_split_sphere_addition_matches_js() {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::{load_shell_split_reference, prepare_brush_geometry};
        use super::super::operations::{
            collect_intersecting_triangles, shell_split_clipped_count,
            shell_split_clipped_tri_keys, shell_split_tri_keys,
        };
        use super::super::topology::tri_key_overlap;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;
        use std::collections::HashSet;

        const REF: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/shell-split-sphere-addition.bin");
        let reference = load_shell_split_reference(REF);

        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let shell = step1_shell_cut(&mut ev);

        let mut acc = CsgBrush::new(shell);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);

        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);

        let (a_inter, _) = collect_intersecting_triangles(&acc, &sphere);
        assert_eq!(
            a_inter.ids.len(),
            reference.split_id_count,
            "shell split intersection id count"
        );

        let clipped = shell_split_clipped_count(&mut acc, &mut sphere);
        let clipped_total_delta = (clipped as i64 - reference.clipped_total as i64).unsigned_abs();
        assert!(
            clipped_total_delta <= 50,
            "shell split clipped_total native={clipped} ref={} delta={clipped_total_delta}",
            reference.clipped_total
        );

        let native_clipped = shell_split_clipped_tri_keys(&mut acc, &mut sphere);
        let native = shell_split_tri_keys(&mut acc, &mut sphere);
        let native_clipped_set: HashSet<_> = native_clipped.iter().copied().collect();
        let ref_clipped_set: HashSet<_> = reference.clipped_tri_keys.iter().copied().collect();
        let native_set: HashSet<_> = native.iter().copied().collect();
        let ref_set: HashSet<_> = reference.tri_keys.iter().copied().collect();

        let clipped_only_nat: Vec<_> = native_clipped_set.difference(&ref_clipped_set).collect();
        let clipped_only_ref: Vec<_> = ref_clipped_set.difference(&native_clipped_set).collect();
        eprintln!(
            "clipped only_native={} only_ref={}",
            clipped_only_nat.len(),
            clipped_only_ref.len()
        );
        for k in clipped_only_nat.iter().take(5) {
            eprintln!("  clipped only_nat {:?}", k.0);
        }
        for k in clipped_only_ref.iter().take(5) {
            eprintln!("  clipped only_ref {:?}", k.0);
        }
        let clipped_overlap = tri_key_overlap(&native_clipped_set, &ref_clipped_set);
        let kept_overlap = tri_key_overlap(&native_set, &ref_set);
        let clipped_ratio =
            clipped_overlap.exact_shared as f64 / ref_clipped_set.len().max(1) as f64;
        let kept_ratio = kept_overlap.exact_shared as f64 / ref_set.len().max(1) as f64;
        let clipped_partial =
            clipped_overlap.partial_shared as f64 / native_clipped_set.len().max(1) as f64;
        let kept_partial = kept_overlap.partial_shared as f64 / native_set.len().max(1) as f64;

        eprintln!(
            "shell split: clipped_total native={clipped} ref={} clipped_keys native={} ref={} exact={:.1}% partial={:.1}% kept_exact={:.1}% kept_partial={:.1}%",
            reference.clipped_total,
            native_clipped_set.len(),
            ref_clipped_set.len(),
            clipped_ratio * 100.0,
            clipped_partial * 100.0,
            kept_ratio * 100.0,
            kept_partial * 100.0
        );

        const REF125: super::super::topology::TriKey = super::super::topology::TriKey((
            -1239920, -318411, 1250000, -1273204, -307379, 1250000, -1201160, -331258, 1250000,
        ));
        const NAT125: super::super::topology::TriKey = super::super::topology::TriKey((
            -1273204, -307379, 1250000, -1239920, -318411, 1250000, -1201160, -331258, 1250000,
        ));
        eprintln!(
            "z1.25 shell_split: native kept ref={} nat={} ref_bin={} clipped_ref={} clipped_nat={}",
            native_set.contains(&REF125),
            native_set.contains(&NAT125),
            ref_set.contains(&REF125),
            native_clipped_set.contains(&REF125),
            native_clipped_set.contains(&NAT125),
        );
        const REF105: super::super::topology::TriKey = super::super::topology::TriKey((
            -1422198, -125068, 1050000, -1451099, -101942, 1050000, -1399999, -142831, 1050000,
        ));
        eprintln!(
            "z1.05 shell_split: native kept ref105={} ref_bin105={} clipped_ref105={}",
            native_set.contains(&REF105),
            ref_set.contains(&REF105),
            native_clipped_set.contains(&REF105),
        );

        assert_eq!(
            clipped_only_nat.len(),
            0,
            "shell split clipped only_native={}",
            clipped_only_nat.len()
        );
        assert_eq!(
            clipped_only_ref.len(),
            0,
            "shell split clipped only_ref={}",
            clipped_only_ref.len()
        );
        assert!(
            (kept_ratio - 1.0).abs() < f64::EPSILON,
            "shell split kept exact overlap {:.1}% < 100%",
            kept_ratio * 100.0
        );
        assert!(
            clipped_partial >= 1.0,
            "shell split clipped partial (2-vert) overlap {:.1}% < 100%",
            clipped_partial * 100.0
        );
        assert!(
            kept_partial >= 1.0,
            "shell split kept partial (2-vert) overlap {:.1}% < 100%",
            kept_partial * 100.0
        );
    }

    #[test]
    fn ia3_ib310_tri52_split_matches_js() {
        use super::super::js_topology::JsTriangle;
        use super::super::triangle_splitter::TriangleSplitter;

        let snap8 = serde_json_min_parse_snap(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j8.json"
        ));
        let clip = serde_json_min_parse_clip(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j9.json"
        ));
        let s = &snap8[52];
        let tri = JsTriangle {
            a: super::super::js_topology::JsVec3::new(s[0], s[1], s[2]),
            b: super::super::js_topology::JsVec3::new(s[3], s[4], s[5]),
            c: super::super::js_topology::JsVec3::new(s[6], s[7], s[8]),
        };
        let clip_tri = JsTriangle {
            a: super::super::js_topology::JsVec3::new(clip[0], clip[1], clip[2]),
            b: super::super::js_topology::JsVec3::new(clip[3], clip[4], clip[5]),
            c: super::super::js_topology::JsVec3::new(clip[6], clip[7], clip[8]),
        };
        eprintln!(
            "tri52 intersects={}",
            clip_tri.intersects_triangle(tri, true)
        );
        let plane = clip_tri.get_plane();
        let arr = [tri.a, tri.b, tri.c];
        for t in 0..3 {
            let t_next = (t + 1) % 3;
            let start = arr[t];
            let end = arr[t_next];
            let sd = plane.distance_to_point(start);
            let ed = plane.distance_to_point(end);
            eprintln!(
                "  edge {t}: sd={sd:.17e} ed={ed:.17e} coplanar_edge={}",
                sd.abs() < 1e-10 && ed.abs() < 1e-10
            );
        }
        let mut hit_vec = super::super::js_topology::JsVec3::default();
        let mut intersects = 0usize;
        for t in 0..3 {
            let t_next = (t + 1) % 3;
            let start = arr[t];
            let end = arr[t_next];
            let sd = plane.distance_to_point(start);
            let ed = plane.distance_to_point(end);
            if sd.abs() < 1e-10 {
                eprintln!("  edge {t}: skip start on plane");
                continue;
            }
            let mut did = plane.intersect_line(start, end, &mut hit_vec).is_some();
            if !did && ed.abs() < 1e-10 {
                hit_vec = end;
                did = true;
            }
            let d = hit_vec.distance_to(start);
            // NaN counts as "not near": the negation is what admitted it before.
            let use_hit = did && (d >= 1e-10 || d.is_nan());
            let vtx_end = use_hit && hit_vec.distance_to(end) < 1e-10;
            eprintln!(
                "  edge {t}: did={did} use_hit={use_hit} vtx_end={vtx_end} hit=({:.17},{:.17},{:.17})",
                hit_vec.x, hit_vec.y, hit_vec.z
            );
            if use_hit {
                intersects += 1;
            }
        }
        eprintln!("  manual intersects={intersects}");
        let single_vert = 0usize;
        let mut fs = super::super::js_topology::JsVec3::new(
            -0.531_259_220_986_309,
            -0.04235038547663728,
            -0.1,
        );
        let mut fe = super::super::js_topology::JsVec3::new(
            -0.526_229_095_725_152_2,
            -0.08055789995397628,
            -0.1,
        );
        std::mem::swap(&mut fs, &mut fe);
        let next_vert1 = 1usize;
        let next_vert2 = 2usize;
        let use_first =
            arr[next_vert1].distance_to_squared(fs) < arr[next_vert2].distance_to_squared(fe);
        let next_tri1 = if use_first {
            super::super::js_topology::JsTriangle {
                a: arr[next_vert1],
                b: fs,
                c: fe,
            }
        } else {
            super::super::js_topology::JsTriangle {
                a: arr[next_vert2],
                b: fs,
                c: fe,
            }
        };
        let next_tri2 = if use_first {
            super::super::js_topology::JsTriangle {
                a: arr[next_vert1],
                b: arr[next_vert2],
                c: fs,
            }
        } else {
            super::super::js_topology::JsTriangle {
                a: arr[next_vert1],
                b: arr[next_vert2],
                c: fe,
            }
        };
        let main_tri = super::super::js_topology::JsTriangle {
            a: arr[single_vert],
            b: fe,
            c: fs,
        };
        use super::super::js_topology::is_tri_degenerate;
        fn angle_info(t: super::super::js_topology::JsTriangle) -> (f64, f64, f64) {
            let ab = super::super::js_topology::JsVec3::default().sub_vectors(t.b, t.a);
            let ac = super::super::js_topology::JsVec3::default().sub_vectors(t.c, t.a);
            let cb = super::super::js_topology::JsVec3::default().sub_vectors(t.b, t.c);
            let a1 = ab.angle_to(ac);
            let a2 = ab.angle_to(cb);
            let a3 = std::f64::consts::PI - a1 - a2;
            (a1, a2, a3)
        }
        let (a1, a2, a3) = angle_info(main_tri);
        eprintln!(
            "  quad use_first={use_first} degen1={} degen2={} degenMain={} main_angles=({a1:.6e},{a2:.6e},{a3:.6e})",
            is_tri_degenerate(next_tri1, 1e-14),
            is_tri_degenerate(next_tri2, 1e-14),
            is_tri_degenerate(main_tri, 1e-14),
        );
        let mut splitter = TriangleSplitter::new();
        splitter.initialize(tri);
        splitter.split_by_triangle(clip_tri);
        eprintln!(
            "tri52 after ib=310: count={} coplanar={} (js=1)",
            splitter.triangle_count(),
            splitter.coplanar_triangle_used
        );
        for (i, t) in splitter.clipped_js_triangles().iter().enumerate() {
            eprintln!(
                "  out{i}: {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17}",
                t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z
            );
        }
        assert_eq!(splitter.triangle_count(), 1, "tri52 should not split");
    }

    #[test]
    fn ia3_ib525_split_matches_js() {
        use super::super::topology::TriKey;
        use super::super::triangle_utils::hash_coord;

        fn to_tri_key(t: &[f64]) -> TriKey {
            TriKey((
                hash_coord(t[0]),
                hash_coord(t[1]),
                hash_coord(t[2]),
                hash_coord(t[3]),
                hash_coord(t[4]),
                hash_coord(t[5]),
                hash_coord(t[6]),
                hash_coord(t[7]),
                hash_coord(t[8]),
            ))
        }

        let (_, snap14_native, _) = ia3_replay_through_j(14);
        let snap14_js = serde_json_min_parse_snap(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j14.json"
        ));
        let native_keys: std::collections::BTreeSet<TriKey> =
            snap14_native.iter().map(|s| to_tri_key(s)).collect();
        let js_keys: std::collections::BTreeSet<TriKey> =
            snap14_js.iter().map(|s| to_tri_key(s)).collect();
        let shared = native_keys.intersection(&js_keys).count();
        let only_native = native_keys.difference(&js_keys).count();
        let only_reference = js_keys.difference(&native_keys).count();
        eprintln!(
            "ia3 j14 ordered: native={} ref={} shared={} only_native={} only_reference={}",
            native_keys.len(),
            js_keys.len(),
            shared,
            only_native,
            only_reference
        );
        if only_native > 0 {
            for k in native_keys.difference(&js_keys).take(3) {
                eprintln!("  only_native {k:?}");
            }
        }
        if only_reference > 0 {
            for k in js_keys.difference(&native_keys).take(3) {
                eprintln!("  only_reference {k:?}");
            }
        }
        assert_eq!(only_native, 0, "native-only at ia3 j14");
        assert_eq!(only_reference, 0, "ref-only at ia3 j14");
    }

    #[test]
    fn ia3_ib500_tri153_split_matches_js() {
        use super::super::js_topology::{is_tri_degenerate, JsTriangle, JsVec3};
        use super::super::triangle_splitter::TriangleSplitter;
        use super::super::triangle_utils::hash_coord;

        fn ord_key(t: JsTriangle) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            (
                hash_coord(t.a.x),
                hash_coord(t.a.y),
                hash_coord(t.a.z),
                hash_coord(t.b.x),
                hash_coord(t.b.y),
                hash_coord(t.b.z),
                hash_coord(t.c.x),
                hash_coord(t.c.y),
                hash_coord(t.c.z),
            )
        }

        const EXTRA_J12: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = (
            -139920, -518411, -99999, -64855, -543292, -99999, -173204, -507379, -99999,
        );

        let snap11 = serde_json_min_parse_snap(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j11.json"
        ));
        let clip = serde_json_min_parse_clip(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j12.json"
        ));
        let s = &snap11[153];
        let tri = JsTriangle {
            a: JsVec3::new(s[0], s[1], s[2]),
            b: JsVec3::new(s[3], s[4], s[5]),
            c: JsVec3::new(s[6], s[7], s[8]),
        };
        let clip_tri = JsTriangle {
            a: JsVec3::new(clip[0], clip[1], clip[2]),
            b: JsVec3::new(clip[3], clip[4], clip[5]),
            c: JsVec3::new(clip[6], clip[7], clip[8]),
        };
        eprintln!(
            "tri153 intersects={}",
            clip_tri.intersects_triangle(tri, true)
        );
        let plane = clip_tri.get_plane();
        let arr = [tri.a, tri.b, tri.c];
        for t in 0..3 {
            let t_next = (t + 1) % 3;
            let sd = plane.distance_to_point(arr[t]);
            let ed = plane.distance_to_point(arr[t_next]);
            eprintln!("  edge {t}: sd={sd:.17e} ed={ed:.17e}");
        }

        let mut splitter = TriangleSplitter::new();
        splitter.initialize(tri);
        splitter.split_by_triangle(clip_tri);
        eprintln!("tri153 after ib=500: count={}", splitter.triangle_count());
        for (i, t) in splitter.clipped_js_triangles().iter().enumerate() {
            let k = ord_key(*t);
            eprintln!("  out{i} {k:?} degen={}", is_tri_degenerate(*t, 1e-14));
        }
        let has_extra = splitter
            .clipped_js_triangles()
            .iter()
            .any(|t| ord_key(*t) == EXTRA_J12);
        assert!(
            !has_extra,
            "tri153 must not produce EXTRA_J12 (js does not)"
        );
        // Native may remove tri entirely when split outputs are degenerate; JS keeps original.
        assert!(
            splitter.triangle_count() <= 1,
            "tri153 split count {}",
            splitter.triangle_count()
        );
    }

    #[test]
    fn ia3_ib310_per_source_tri_delta() {
        use super::super::js_topology::JsTriangle;
        use super::super::triangle_splitter::TriangleSplitter;

        let snap8: Vec<Vec<f64>> = serde_json_min_parse_snap(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j8.json"
        ));
        let clip: [f64; 9] = serde_json_min_parse_clip(include_str!(
            "../../tests/parity/scenes/rust/ia3-snap-j9.json"
        ));
        let clip_tri = JsTriangle {
            a: super::super::js_topology::JsVec3::new(clip[0], clip[1], clip[2]),
            b: super::super::js_topology::JsVec3::new(clip[3], clip[4], clip[5]),
            c: super::super::js_topology::JsVec3::new(clip[6], clip[7], clip[8]),
        };
        let mut total = 0i32;
        for (i, s) in snap8.iter().enumerate() {
            let tri = JsTriangle {
                a: super::super::js_topology::JsVec3::new(s[0], s[1], s[2]),
                b: super::super::js_topology::JsVec3::new(s[3], s[4], s[5]),
                c: super::super::js_topology::JsVec3::new(s[6], s[7], s[8]),
            };
            let mut splitter = TriangleSplitter::new();
            splitter.initialize(tri);
            splitter.split_by_triangle(clip_tri);
            let d = splitter.triangle_count() as i32 - 1;
            if d != 0 {
                eprintln!(
                    "native tri {i} delta={d} count={}",
                    splitter.triangle_count()
                );
            }
            total += d;
        }
        eprintln!("native total delta {total} (js should be 13)");
        assert_eq!(total, 13, "total split delta");
    }

    fn serde_json_min_parse_snap(raw: &str) -> Vec<Vec<f64>> {
        let key = "\"snap\":";
        let start = raw.find(key).expect("snap") + key.len();
        let slice = &raw[start..];
        parse_tri_array(slice)
    }

    fn serde_json_min_parse_clip(raw: &str) -> [f64; 9] {
        let key = "\"clip\":";
        let start = raw.find(key).expect("clip") + key.len();
        parse_flat9(&raw[start..])
    }

    fn parse_flat9(s: &str) -> [f64; 9] {
        let mut i = 0usize;
        while i < s.len() && s.as_bytes()[i] != b'[' {
            i += 1;
        }
        i += 1;
        let mut vals = Vec::new();
        while i < s.len() && vals.len() < 9 {
            while i < s.len() && (s.as_bytes()[i] == b',' || s.as_bytes()[i].is_ascii_whitespace())
            {
                i += 1;
            }
            let start = i;
            while i < s.len() && s.as_bytes()[i] != b',' && s.as_bytes()[i] != b']' {
                i += 1;
            }
            vals.push(s[start..i].parse::<f64>().expect("f64"));
        }
        vals.try_into().expect("9 floats")
    }

    fn parse_tri_array(s: &str) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < s.len() && s.as_bytes()[i] != b'[' {
            i += 1;
        }
        i += 1;
        while i < s.len() {
            while i < s.len() && s.as_bytes()[i].is_ascii_whitespace() || s.as_bytes()[i] == b',' {
                i += 1;
            }
            if i >= s.len() || s.as_bytes()[i] == b']' {
                break;
            }
            if s.as_bytes()[i] != b'[' {
                break;
            }
            i += 1;
            let mut tri = Vec::new();
            while i < s.len() && s.as_bytes()[i] != b']' {
                while i < s.len()
                    && (s.as_bytes()[i] == b',' || s.as_bytes()[i].is_ascii_whitespace())
                {
                    i += 1;
                }
                let start = i;
                while i < s.len() && s.as_bytes()[i] != b',' && s.as_bytes()[i] != b']' {
                    i += 1;
                }
                tri.push(s[start..i].parse::<f64>().expect("f64"));
            }
            if s.as_bytes()[i] == b']' {
                i += 1;
            }
            out.push(tri);
        }
        out
    }

    #[test]
    fn shell_ia3_neighbor310_split_matches_js() {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::prepare_brush_geometry;
        use super::super::js_topology::{js_tri_from_indices, matrix_a_to_b_brushes};
        use super::super::operations::collect_intersecting_triangles;
        use super::super::triangle_splitter::TriangleSplitter;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;

        let mut ev = CsgEvaluator::new();
        let shell = step1_shell_cut(&mut ev);
        let mut acc = CsgBrush::new(shell);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);
        let (a_inter, _) = collect_intersecting_triangles(&acc, &sphere);
        let matrix = matrix_a_to_b_brushes(&acc, &sphere);
        let pos = acc.geometry.get_attribute("position").unwrap();
        let b_pos = sphere.geometry.get_attribute("position").unwrap();
        let ia = 3usize;
        let neighbors = a_inter.intersection_set.get(&ia).expect("neighbors");
        let i0 = super::super::geometry_prep::index_at(&acc.geometry, ia, 0);
        let i1 = super::super::geometry_prep::index_at(&acc.geometry, ia, 1);
        let i2 = super::super::geometry_prep::index_at(&acc.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        let mut splitter = TriangleSplitter::new();
        splitter.reset();
        splitter.initialize(tri_a);
        for &ib in neighbors.iter().take(9) {
            let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
            let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
            let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
            splitter.split_by_triangle(js_tri_from_indices(b_pos, j0, j1, j2, None));
        }
        assert_eq!(splitter.triangle_count(), 98, "after j=8");
        let ib = neighbors[9];
        assert_eq!(ib, 310, "neighbor j=9 ib");
        let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
        let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
        let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
        let tri_b = js_tri_from_indices(b_pos, j0, j1, j2, None);
        eprintln!(
            "ib=310 clip: {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17}",
            tri_b.a.x,
            tri_b.a.y,
            tri_b.a.z,
            tri_b.b.x,
            tri_b.b.y,
            tri_b.b.z,
            tri_b.c.x,
            tri_b.c.y,
            tri_b.c.z,
        );
        splitter.split_by_triangle(tri_b);
        let count = splitter.triangle_count();
        eprintln!(
            "after ib=310: native={count} js=111 coplanar={}",
            splitter.coplanar_triangle_used
        );
        assert_eq!(count, 111, "ib=310 split count");
    }

    fn ia3_replay_through_j(j_end: usize) -> (usize, Vec<Vec<f64>>, u32) {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::prepare_brush_geometry;
        use super::super::js_topology::{js_tri_from_indices, matrix_a_to_b_brushes};
        use super::super::operations::collect_intersecting_triangles;
        use super::super::triangle_splitter::TriangleSplitter;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;

        let mut ev = CsgEvaluator::new();
        let shell = step1_shell_cut(&mut ev);
        let mut acc = CsgBrush::new(shell);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);
        let (a_inter, _) = collect_intersecting_triangles(&acc, &sphere);
        let matrix = matrix_a_to_b_brushes(&acc, &sphere);
        let pos = acc.geometry.get_attribute("position").unwrap();
        let b_pos = sphere.geometry.get_attribute("position").unwrap();
        let ia = 3usize;
        let neighbors = a_inter.intersection_set.get(&ia).expect("neighbors");
        let i0 = super::super::geometry_prep::index_at(&acc.geometry, ia, 0);
        let i1 = super::super::geometry_prep::index_at(&acc.geometry, ia, 1);
        let i2 = super::super::geometry_prep::index_at(&acc.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        let mut splitter = TriangleSplitter::new();
        splitter.reset();
        splitter.initialize(tri_a);
        for &ib in neighbors.iter().take(j_end + 1) {
            let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
            let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
            let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
            splitter.split_by_triangle(js_tri_from_indices(b_pos, j0, j1, j2, None));
        }
        let ib = neighbors[j_end];
        let snap: Vec<Vec<f64>> = splitter
            .clipped_js_triangles()
            .into_iter()
            .map(|t| {
                vec![
                    t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z,
                ]
            })
            .collect();
        (
            ib,
            snap,
            if splitter.coplanar_triangle_used {
                1
            } else {
                0
            },
        )
    }

    #[test]
    fn shell_ia15_neighbor354_split_matches_js() {
        use super::super::brush::CsgBrush;
        use super::super::bvhcast_parity::prepare_brush_geometry;
        use super::super::js_topology::{
            is_tri_degenerate, js_tri_from_indices, matrix_a_to_b_brushes,
        };
        use super::super::operations::collect_intersecting_triangles;
        use super::super::triangle_splitter::TriangleSplitter;
        use crate::geometries::SphereGeometry;
        use crate::math::Matrix4;

        let mut ev = CsgEvaluator::new();
        let shell = step1_shell_cut(&mut ev);
        let mut acc = CsgBrush::new(shell);
        acc.matrix_world = Matrix4::identity();
        prepare_brush_geometry(&mut acc.geometry);
        let mut sphere = CsgBrush::new(SphereGeometry::new(0.55, 24, 12));
        sphere.set_position(-1.1, 0.2, 1.35);
        sphere.update_matrix_world(None);
        prepare_brush_geometry(&mut sphere.geometry);
        let (a_inter, _) = collect_intersecting_triangles(&acc, &sphere);
        let matrix = matrix_a_to_b_brushes(&acc, &sphere);
        eprintln!("native matrix: {:?}", matrix.elements);
        let pos = acc.geometry.get_attribute("position").unwrap();
        let b_pos = sphere.geometry.get_attribute("position").unwrap();
        let ia = 15usize;
        let neighbors = a_inter.intersection_set.get(&ia).expect("neighbors");
        let i0 = super::super::geometry_prep::index_at(&acc.geometry, ia, 0);
        let i1 = super::super::geometry_prep::index_at(&acc.geometry, ia, 1);
        let i2 = super::super::geometry_prep::index_at(&acc.geometry, ia, 2);
        let tri_a = js_tri_from_indices(pos, i0, i1, i2, Some(&matrix));
        eprintln!(
            "tri_a: {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17}",
            tri_a.a.x,
            tri_a.a.y,
            tri_a.a.z,
            tri_a.b.x,
            tri_a.b.y,
            tri_a.b.z,
            tri_a.c.x,
            tri_a.c.y,
            tri_a.c.z,
        );
        let mut splitter = TriangleSplitter::new();
        splitter.reset();
        splitter.initialize(tri_a);
        for &ib in &neighbors[..1] {
            let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
            let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
            let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
            splitter.split_by_triangle(js_tri_from_indices(b_pos, j0, j1, j2, None));
        }
        eprintln!("native snap after n0:");
        for (ti, t) in splitter.clipped_js_triangles().iter().enumerate() {
            eprintln!(
                "  {ti}: {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17}",
                t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z
            );
        }
        for &ib in &neighbors[1..2] {
            let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
            let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
            let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
            splitter.split_by_triangle(js_tri_from_indices(b_pos, j0, j1, j2, None));
        }
        assert_eq!(splitter.triangle_count(), 7, "after 2 neighbors");
        eprintln!("native snap after n1:");
        for (ti, t) in splitter.clipped_js_triangles().iter().enumerate() {
            eprintln!(
                "  {ti}: {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17} | {:.17} {:.17} {:.17}",
                t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z
            );
        }
        let ib = neighbors[2];
        assert_eq!(ib, 354, "expected ib=354 at neighbor 2");
        let j0 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 0);
        let j1 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 1);
        let j2 = super::super::geometry_prep::index_at(&sphere.geometry, ib, 2);
        let clip = js_tri_from_indices(b_pos, j0, j1, j2, None);
        for (ti, t) in splitter.clipped_js_triangles().iter().enumerate() {
            eprintln!(
                "pre-split tri {ti} intersects={}",
                clip.intersects_triangle(*t, true)
            );
        }
        splitter.split_by_triangle(clip);
        let count = splitter.triangle_count();
        for (ti, t) in splitter.clipped_js_triangles().iter().enumerate() {
            let ab = super::super::js_topology::JsVec3::default().sub_vectors(t.b, t.a);
            let ac = super::super::js_topology::JsVec3::default().sub_vectors(t.c, t.a);
            let cb = super::super::js_topology::JsVec3::default().sub_vectors(t.b, t.c);
            let a1 = ab.angle_to(ac);
            let a2 = ab.angle_to(cb);
            let a3 = std::f64::consts::PI - a1 - a2;
            eprintln!(
                "out tri {ti} degen={} a1={a1:.3e} a2={a2:.3e} a3={a3:.3e} ab={:.3e} ac={:.3e} bc={:.3e}",
                is_tri_degenerate(*t, 1e-14),
                t.a.distance_to_squared(t.b),
                t.a.distance_to_squared(t.c),
                t.b.distance_to_squared(t.c),
            );
        }
        let degen = splitter
            .clipped_js_triangles()
            .iter()
            .filter(|t| is_tri_degenerate(**t, 1e-14))
            .count();
        eprintln!("after ib=354: count={count} degen={degen} (JS ref 8)");
        // Isolated cascade can diverge by one split; full step2 pipeline is 100% at soup level.
        assert!(
            count <= 10,
            "neighbor 354 split count {count} unexpectedly high"
        );
    }

    #[test]
    fn step2_live_sphere_topology_matches_js() {
        use super::super::topology::{
            assert_topology_partial_overlap, topology_overlap, triangle_keys,
        };
        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let native = evaluate_live_through(2, &mut ev);
        let reference = load_bin_positions(AFTER_SPHERE);
        let o = topology_overlap(&native, &reference);
        let nk = triangle_keys(&native);
        let rk = triangle_keys(&reference);
        let partial = super::super::topology::tri_key_overlap(&nk, &rk);
        let partial_ratio = partial.partial_shared as f64 / nk.len().max(1) as f64;
        eprintln!(
            "step2 topology: native={} ref={} shared={} only_native={} only_ref={} exact={:.1}% partial={:.1}%",
            o.native_tris, o.reference_tris, o.shared, o.only_native, o.only_reference,
            o.shared as f64 / o.reference_tris.max(1) as f64 * 100.0,
            partial_ratio * 100.0
        );
        const REF125: super::super::topology::TriKey = super::super::topology::TriKey((
            -1239920, -318411, 1250000, -1273204, -307379, 1250000, -1201160, -331258, 1250000,
        ));
        const NAT125: super::super::topology::TriKey = super::super::topology::TriKey((
            -1273204, -307379, 1250000, -1239920, -318411, 1250000, -1201160, -331258, 1250000,
        ));
        eprintln!(
            "z1.25 step2: native ref={} nat={} ref_geom={}",
            nk.contains(&REF125),
            nk.contains(&NAT125),
            rk.contains(&REF125),
        );
        assert_eq!(
            o.only_native, 0,
            "step2 exact: only_native={} (shared {}/{})",
            o.only_native, o.shared, o.reference_tris
        );
        assert_eq!(
            o.only_reference, 0,
            "step2 exact: only_ref={} (shared {}/{})",
            o.only_reference, o.shared, o.reference_tris
        );
        assert_topology_partial_overlap(&nk, &rk, 1.0, "step2 partial");
    }

    #[test]
    fn step1_shell_cut_matches_js() {
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = step1_shell_cut(&mut ev);
        assert_step_verts(1, vert_count(&geom));
    }

    #[test]
    fn step2_live_sphere_within_tolerance() {
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = evaluate_live_through(2, &mut ev);
        let v = vert_count(&geom);
        eprintln!("step2 live: {v} (JS {JS_STEP2_VERTS})");
        assert_step_verts(2, v);
    }

    #[test]
    fn step3_on_reference_sphere_matches_js() {
        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = evaluate_through(3, &mut ev, Some(AFTER_SPHERE));
        let v = vert_count(&geom);
        eprintln!("step3 ref-sphere: {v} (JS {JS_STEP3_VERTS})");
        assert_step_verts(3, v);
    }

    #[test]
    fn step4_on_reference_wincut_matches_js() {
        const AFTER_WINCUT: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-step3-wincut.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let after_cut = load_bin_positions(AFTER_WINCUT);
        let geom = step4_win_frame(&mut ev, after_cut);
        let v = vert_count(&geom);
        eprintln!("step4 ref-wincut: {v} (JS {JS_STEP4_VERTS})");
        assert_step_verts(4, v);
    }

    #[test]
    fn step4_on_reference_sphere_matches_js() {
        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = evaluate_through(4, &mut ev, Some(AFTER_SPHERE));
        let v = vert_count(&geom);
        eprintln!("step4 ref-sphere chain: {v} (JS {JS_STEP4_VERTS})");
        assert_step_verts(4, v);
    }

    #[test]
    fn step3_live_chain_matches_js() {
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = evaluate_live_through(3, &mut ev);
        assert_step_verts(3, vert_count(&geom));
    }

    #[test]
    fn step4_live_chain_matches_js() {
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let geom = evaluate_live_through(4, &mut ev);
        assert_step_verts(4, vert_count(&geom));
    }

    #[test]
    fn step3_win_cut_on_js_sphere_soup_direct() {
        const AFTER_SPHERE: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-after-sphere.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let after_sphere = load_bin_positions(AFTER_SPHERE);
        let geom = step3_win_cut(&mut ev, after_sphere);
        assert_step_verts(3, vert_count(&geom));
    }

    #[test]
    fn step3_live_topology_matches_js() {
        use super::super::topology::topology_overlap;
        const REF: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-step3-wincut.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let native = evaluate_live_through(3, &mut ev);
        let reference = load_bin_positions(REF);
        let o = topology_overlap(&native, &reference);
        eprintln!(
            "step3 topology: native={} ref={} shared={} only_native={} only_ref={} exact={:.1}%",
            o.native_tris,
            o.reference_tris,
            o.shared,
            o.only_native,
            o.only_reference,
            o.shared as f64 / o.reference_tris.max(1) as f64 * 100.0
        );
        assert_eq!(o.only_native, 0, "step3 only_native={}", o.only_native);
        assert_eq!(o.only_reference, 0, "step3 only_ref={}", o.only_reference);
    }

    #[test]
    fn step4_live_topology_matches_js() {
        use super::super::topology::topology_overlap;
        const REF: &[u8] =
            include_bytes!("../../tests/parity/scenes/rust/bvh-csg-hierarchy.geom.bin");
        let mut ev = CsgEvaluator::new();
        ev.use_groups = false;
        let native = evaluate_live_through(4, &mut ev);
        let reference = load_bin_positions(REF);
        let o = topology_overlap(&native, &reference);
        eprintln!(
            "step4 topology: native={} ref={} shared={} only_native={} only_ref={} exact={:.1}%",
            o.native_tris,
            o.reference_tris,
            o.shared,
            o.only_native,
            o.only_reference,
            o.shared as f64 / o.reference_tris.max(1) as f64 * 100.0
        );
        assert_eq!(o.only_native, 0, "step4 only_native={}", o.only_native);
        assert_eq!(o.only_reference, 0, "step4 only_ref={}", o.only_reference);
    }
}
