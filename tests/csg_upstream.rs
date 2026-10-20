//! The CSG port against three-bvh-csg itself.
//!
//! `tests/parity/scenes/rust/bvh-csg-upstream.txt` is three-bvh-csg@0.0.16 run
//! in Node on the bvh-csg parity scenes (`tests/parity/export-bvh-csg-upstream.mjs`):
//! every `Evaluator.evaluate` call, its two operands as they went in, and what
//! upstream produced. Each is replayed here through [`CsgEvaluator`] on the
//! same operands and must come out the same: the same number of triangles —
//! which needs three-mesh-bvh's tree, its traversal order, and three.js's
//! f64 arithmetic down to the operand order, since a face is cut in the
//! order the BVH meets its cutting triangles and a vertex on another face is
//! classified by the last bit — and the same surface area and volume.

use threers::core::{BufferAttribute, BufferGeometry, GeometryGroup};
use threers::csg::{CsgBrush, CsgEvaluator};

const FIXTURE: &str = include_str!("parity/scenes/rust/bvh-csg-upstream.txt");

fn base64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c) << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    out
}

struct Brush {
    geometry: BufferGeometry,
    matrix: [f64; 16],
}

struct Expected {
    triangles: usize,
    area: f64,
    volume: f64,
}

struct Call {
    scene: String,
    operations: Vec<u8>,
    use_groups: bool,
    attributes: Vec<String>,
    brushes: Vec<Brush>,
    results: Vec<Expected>,
}

fn parse() -> Vec<Call> {
    let mut calls = Vec::new();
    let mut call: Option<Call> = None;
    for line in FIXTURE.lines() {
        let mut f = line.split(' ');
        match f.next() {
            Some("call") => {
                let scene = f.next().unwrap().to_string();
                let operations = f
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|v| v.parse().unwrap())
                    .collect();
                let use_groups = f.next().unwrap() == "1";
                let attributes = f.next().unwrap().split(',').map(String::from).collect();
                call = Some(Call {
                    scene,
                    operations,
                    use_groups,
                    attributes,
                    brushes: vec![],
                    results: vec![],
                });
            }
            Some("brush") => call.as_mut().unwrap().brushes.push(Brush {
                geometry: BufferGeometry::new(),
                matrix: [0.0; 16],
            }),
            Some("matrix") => {
                let b = call.as_mut().unwrap().brushes.last_mut().unwrap();
                for (i, v) in f.enumerate() {
                    b.matrix[i] = v.parse().unwrap();
                }
            }
            Some("group") => {
                let v: Vec<i64> = f.map(|v| v.parse().unwrap()).collect();
                let b = call.as_mut().unwrap().brushes.last_mut().unwrap();
                b.geometry.groups.push(GeometryGroup {
                    start: v[0] as u32,
                    count: if v[1] < 0 { u32::MAX } else { v[1] as u32 },
                    material_index: v[2] as u32,
                });
            }
            Some("attr") => {
                let name = f.next().unwrap();
                let size: usize = f.next().unwrap().parse().unwrap();
                let data: Vec<f32> = base64(f.next().unwrap())
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let b = call.as_mut().unwrap().brushes.last_mut().unwrap();
                b.geometry
                    .set_attribute(name, BufferAttribute::new(data, size));
            }
            Some("result") => {
                let v: Vec<&str> = f.collect();
                call.as_mut().unwrap().results.push(Expected {
                    triangles: v[0].parse().unwrap(),
                    area: v[1].parse().unwrap(),
                    volume: v[2].parse().unwrap(),
                });
            }
            Some("end") => calls.push(call.take().unwrap()),
            _ => {}
        }
    }
    calls
}

/// Triangle count, area and signed volume of a set of non-indexed groups.
fn measure(groups: &[Vec<(String, usize, Vec<f32>)>]) -> (usize, f64, f64) {
    let (mut tris, mut area, mut volume) = (0usize, 0.0f64, 0.0f64);
    for group in groups {
        let pos = &group.iter().find(|(n, _, _)| n == "position").unwrap().2;
        for t in pos.chunks_exact(9) {
            let p = |i: usize| [t[i] as f64, t[i + 1] as f64, t[i + 2] as f64];
            let (a, b, c) = (p(0), p(3), p(6));
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            area += (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt() / 2.0;
            let bxc = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            volume += (a[0] * bxc[0] + a[1] * bxc[1] + a[2] * bxc[2]) / 6.0;
            tris += 1;
        }
    }
    (tris, area, volume)
}

#[test]
fn every_evaluation_encloses_what_three_bvh_csg_does() {
    let calls = parse();
    assert!(
        calls.len() >= 17,
        "fixture holds {} evaluations",
        calls.len()
    );
    let mut failures = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        let mut brushes = call.brushes.iter().map(|b| {
            let mut brush = CsgBrush::new(b.geometry.clone());
            brush.set_matrix_world(b.matrix);
            brush
        });
        let (mut a, mut b) = (brushes.next().unwrap(), brushes.next().unwrap());
        let mut evaluator = CsgEvaluator::new();
        evaluator.use_groups = call.use_groups;
        let names: Vec<&str> = call.attributes.iter().map(String::as_str).collect();
        let results = evaluator
            .evaluate_groups(&mut a, &mut b, &call.operations, &names)
            .unwrap();
        for (op, (got, want)) in results.iter().zip(&call.results).enumerate() {
            let (tris, area, volume) = measure(got);
            // f32 output summed in f64 here and in f64 upstream: agreement to
            // a part in a million. A volume near zero (an open, hollow result)
            // is compared against its area-sized scale.
            let area_err = (area - want.area).abs() / want.area.max(1e-9);
            let volume_err = (volume - want.volume).abs() / want.volume.abs().max(want.area * 1e-2);
            let tri_ok = tris == want.triangles;
            let line = format!(
                "{} #{i} op {}: {tris} tris (upstream {}), area {area:.6} ({:.6}), volume {volume:.6} ({:.6})",
                call.scene, call.operations[op], want.triangles, want.area, want.volume
            );
            eprintln!("{line}");
            if area_err > 1e-6 || volume_err > 1e-6 || !tri_ok {
                failures.push(line);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} evaluations differ from three-bvh-csg:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The operands three.js built for the union scene are bit for bit the ones
/// threers builds: `SphereGeometry(1, 32, 16)` and `BoxGeometry(1.2, 1.2,
/// 1.2)`, as triangle soups. Vertices a hair off an axis decide which side of
/// a ray or cutting plane they fall on, so near is not enough.
#[test]
fn threers_primitives_are_three_js_bit_for_bit() {
    use threers::geometries::{BoxGeometry, SphereGeometry};
    let calls = parse();
    let union = calls
        .iter()
        .find(|c| c.scene == "bvh-csg-union")
        .expect("union call");
    for (brush, ours) in union.brushes.iter().zip([
        SphereGeometry::new(1.0, 32, 16).to_non_indexed(),
        BoxGeometry::new(1.2, 1.2, 1.2).to_non_indexed(),
    ]) {
        for name in ["position", "normal", "uv"] {
            let want = &brush.geometry.get_attribute(name).unwrap().array;
            let got = &ours.get_attribute(name).unwrap().array;
            assert_eq!(got.len(), want.len(), "{name} length");
            let differ = got
                .iter()
                .zip(want.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differ,
                0,
                "{name}: {differ} of {} values differ from three.js",
                want.len()
            );
        }
    }
}

/// `evaluate_hierarchy` on the hierarchy scene, built here from threers's own
/// primitives, ends where three-bvh-csg's `evaluateHierarchy` does: the same
/// triangles, area and volume as upstream's final evaluation.
#[test]
fn evaluate_hierarchy_ends_where_three_bvh_csg_does() {
    use threers::csg::{
        evaluate_hierarchy, CsgNode, CsgOperation, CsgOperationGroup, ADDITION, SUBTRACTION,
    };
    use threers::geometries::{BoxGeometry, SphereGeometry};

    let soup = |g: BufferGeometry| g.to_non_indexed();
    let mut root = CsgOperation::new(soup(BoxGeometry::new(4.0, 2.5, 2.5)), ADDITION);
    let cut = CsgOperation::new(soup(BoxGeometry::new(3.6, 2.1, 2.1)), SUBTRACTION);
    let mut sphere = CsgOperation::new(soup(SphereGeometry::new(0.55, 24, 12)), ADDITION);
    sphere.set_position(-1.1, 0.2, 1.35);
    let mut window = CsgOperationGroup::new();
    window.add_child(CsgNode::Operation(CsgOperation::new(
        soup(BoxGeometry::new(1.2, 1.0, 0.5)),
        SUBTRACTION,
    )));
    window.add_child(CsgNode::Operation(CsgOperation::new(
        soup(BoxGeometry::new(1.2, 1.0, 0.12)),
        ADDITION,
    )));
    window.set_position(0.8, 0.15, 1.35);
    root.add_child(CsgNode::Operation(cut));
    root.add_child(CsgNode::Operation(sphere));
    root.add_child(CsgNode::Group(window));

    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = false;
    let result = evaluate_hierarchy(&mut root, &mut evaluator);
    let positions = result.get_attribute("position").unwrap().array.clone();
    let (tris, area, volume) = measure(&[vec![("position".to_string(), 3, positions)]]);

    let calls = parse();
    let last = calls
        .iter()
        .rfind(|c| c.scene == "bvh-csg-hierarchy")
        .expect("hierarchy calls");
    let want = &last.results[0];
    assert_eq!(tris, want.triangles, "triangles");
    assert!(
        (area - want.area).abs() / want.area < 1e-6,
        "area {area} vs {}",
        want.area
    );
    assert!(
        (volume - want.volume).abs() / want.volume < 1e-6,
        "volume {volume} vs {}",
        want.volume
    );
}
