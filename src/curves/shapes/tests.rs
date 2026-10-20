//! Parity with three.js r165, against buffers three.js itself produced.
//!
//! `testdata/r165.json` was written by running three.js 0.165 in node on the
//! shapes built below (the same calls in JavaScript). Small buffers are stored
//! whole; large ones as their length, an FNV-1a hash of their `f32` bits and
//! their first 48 values. Comparisons are exact: the point is bit parity.

use super::*;
use crate::core::BufferGeometry;
use crate::geometries::{
    ExtrudeGeometry, ExtrudeOptions, ExtrudePath, ShapeGeometry, TextGeometry, TextOptions,
};
use crate::loaders::json::{self, Value};
use crate::loaders::typeface::Font;

const FIXTURE: &str = include_str!("testdata/r165.json");
const FONT: &str = include_str!("testdata/helvetiker_subset.typeface.json");

fn fixture() -> Value {
    json::parse(FIXTURE).expect("fixture parses")
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_number().expect("number"))
        .collect()
}

fn v(x: f64, y: f64) -> Vec2d {
    Vec2d::new(x, y)
}

fn flat(points: &[Vec2d]) -> Vec<f64> {
    points.iter().flat_map(|p| [p.x, p.y]).collect()
}

/// The shape every test below shares: each curve type, a rotated clockwise
/// ellipse, an explicit close, a circular hole and a triangular one.
fn mixed() -> Shape {
    let mut s = Shape::new();
    s.move_to(0.0, 0.0);
    s.line_to(2.0, 0.0);
    s.quadratic_curve_to(3.0, 0.5, 2.5, 1.5);
    s.bezier_curve_to(2.2, 2.5, 1.0, 2.8, 0.5, 2.0);
    s.spline_thru(&[v(0.2, 1.6), v(-0.4, 1.2), v(-0.3, 0.6)]);
    s.absellipse(-0.5, 0.3, 0.4, 0.25, 0.3, 4.0, true, 0.2);
    s.close_path();
    let mut hole = Path::new();
    hole.absarc(1.2, 1.0, 0.35, 0.0, std::f64::consts::PI * 2.0, false);
    s.holes.push(hole);
    let mut hole2 = Path::new();
    hole2.move_to(1.8, 0.3);
    hole2.line_to(2.1, 0.3);
    hole2.line_to(1.95, 0.6);
    s.holes.push(hole2);
    s
}

fn triangle() -> Shape {
    Shape::from_points(&[v(0.0, 0.5), v(-0.5, -0.5), v(0.5, -0.5)])
}

fn fnv(values: &[f32]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for f in values {
        for b in f.to_bits().to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
    }
    h
}

fn attr<'a>(g: &'a BufferGeometry, name: &str) -> &'a [f32] {
    &g.get_attribute(name)
        .unwrap_or_else(|| panic!("no {name}"))
        .array
}

fn groups_of(g: &BufferGeometry) -> Vec<f64> {
    g.groups
        .iter()
        .flat_map(|x| [x.start as f64, x.count as f64, x.material_index as f64])
        .collect()
}

fn expected_groups(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().flat_map(nums).collect()
}

fn assert_exact(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        // JSON has no -0, so zero matches either sign here; the node suite
        // (crates/threers-js/test/shapes-parity.test.mjs) checks signs too.
        assert!(
            g.to_bits() == (w as f32).to_bits() || (w == 0.0 && g == 0.0),
            "{name}[{i}]: got {g}, three.js {w}"
        );
    }
}

fn assert_index(g: &BufferGeometry, want: &Value) {
    match want {
        Value::Null => assert!(g.index.is_none(), "expected no index"),
        w => {
            let got: Vec<f64> = g
                .index
                .as_ref()
                .expect("index")
                .iter()
                .map(|&i| i as f64)
                .collect();
            assert_eq!(got, nums(w), "index");
        }
    }
}

/// Whole buffers.
fn assert_geometry(name: &str, g: &BufferGeometry, want: &Value) {
    for a in ["position", "normal", "uv"] {
        assert_exact(
            &format!("{name}.{a}"),
            attr(g, a),
            &nums(want.get(a).unwrap()),
        );
    }
    assert_index(g, want.get("index").unwrap());
    assert_eq!(
        groups_of(g),
        expected_groups(want.get("groups").unwrap()),
        "{name}: groups"
    );
}

/// Length, hash and head.
fn assert_digest(name: &str, g: &BufferGeometry, want: &Value) {
    for a in ["position", "normal", "uv"] {
        let d = want.get(a).unwrap();
        let got = attr(g, a);
        let head = nums(d.get("head").unwrap());
        assert_exact(
            &format!("{name}.{a} (head)"),
            &got[..head.len().min(got.len())],
            &head,
        );
        assert_eq!(
            got.len() as f64,
            d.get("len").unwrap().as_number().unwrap(),
            "{name}.{a}: length"
        );
        assert_eq!(
            fnv(got) as f64,
            d.get("fnv").unwrap().as_number().unwrap(),
            "{name}.{a}: hash"
        );
    }
    assert_index(g, want.get("index").unwrap());
    assert_eq!(
        groups_of(g),
        expected_groups(want.get("groups").unwrap()),
        "{name}: groups"
    );
}

fn tri_list(t: &[[u32; 3]]) -> Vec<f64> {
    t.iter().flatten().map(|&i| i as f64).collect()
}

#[test]
fn earcut_matches_three() {
    let f = fixture();
    let mut sq = vec![v(10.0, 0.0), v(0.0, 50.0), v(60.0, 60.0), v(70.0, 10.0)];
    assert_eq!(
        tri_list(&shape_utils::triangulate_shape(&mut sq, &mut [])),
        nums(f.get("earcut_square").unwrap())
    );
    let mut outer = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0), v(0.0, 100.0)];
    let mut holes = vec![vec![
        v(20.0, 20.0),
        v(80.0, 20.0),
        v(80.0, 80.0),
        v(20.0, 80.0),
    ]];
    assert_eq!(
        tri_list(&shape_utils::triangulate_shape(&mut outer, &mut holes)),
        nums(f.get("earcut_hole").unwrap())
    );
    // Over 80 vertices: the z-order hashed path.
    let pi = std::f64::consts::PI;
    let mut big: Vec<Vec2d> = (0..120)
        .map(|i| {
            let a = -(i as f64) / 120.0 * pi * 2.0;
            let r = 10.0 + 3.0 * (a * 7.0).sin();
            v(r * a.cos(), r * a.sin())
        })
        .collect();
    let mut h1 = Vec::new();
    let mut h2 = Vec::new();
    for i in 0..9 {
        let a = i as f64 / 9.0 * pi * 2.0;
        h1.push(v(-3.0 + 1.5 * a.cos(), 1.5 * a.sin()));
        h2.push(v(3.0 + 1.2 * a.cos(), 0.5 + 1.2 * a.sin()));
    }
    assert_eq!(
        tri_list(&shape_utils::triangulate_shape(&mut big, &mut [h1, h2])),
        nums(f.get("earcut_big").unwrap())
    );
}

/// Equal to the last few bits. Sampling goes through `sin` and `cos`, which
/// come from the platform's maths library: macOS's agrees with V8's (which
/// made the fixture) bit for bit, glibc's is a bit off on the odd value.
fn assert_same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "lengths");
    for (k, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
            "[{k}]: {a} != {b}"
        );
    }
}

#[test]
fn path_sampling_matches_three() {
    let f = fixture();
    let m = mixed();
    assert_same(
        &flat(&m.get_points(5)),
        &nums(f.get("mixed_points_5").unwrap()),
    );
    assert_same(
        &flat(&m.get_points(12)),
        &nums(f.get("mixed_points_12").unwrap()),
    );
    assert_same(
        &flat(&m.get_spaced_points(7)),
        &nums(f.get("mixed_spaced_7").unwrap()),
    );
    assert_same(
        &[m.get_length()],
        &[f.get("mixed_length").unwrap().as_number().unwrap()],
    );
    let at = [m.get_point_at(0.37).unwrap(), m.get_point_at(0.81).unwrap()];
    assert_same(&flat(&at), &nums(f.get("mixed_point_at").unwrap()));
    assert_same(
        &flat(&m.holes[0].get_points(4)),
        &nums(f.get("hole_points_4").unwrap()),
    );

    let mut ac = Path::new();
    ac.move_to(0.0, 0.0);
    ac.line_to(1.0, 0.0);
    ac.line_to(1.0, 1.0);
    ac.auto_close = true;
    assert_same(
        &flat(&ac.get_points(12)),
        &nums(f.get("autoclose_points").unwrap()),
    );
    assert_same(
        &flat(&ac.get_spaced_points(4)),
        &nums(f.get("autoclose_spaced").unwrap()),
    );
}

#[test]
fn shape_geometry_matches_three() {
    let f = fixture();
    let multi = ShapeGeometry::from_shapes(&[mixed(), triangle()], 4);
    assert_geometry("shapegeom_multi", &multi, f.get("shapegeom_multi").unwrap());
    let single = ShapeGeometry::new(&triangle(), 12);
    assert_geometry(
        "shapegeom_single",
        &single,
        f.get("shapegeom_single").unwrap(),
    );
}

#[test]
fn extrude_geometry_matches_three() {
    let f = fixture();
    let no_bevel = ExtrudeGeometry::from_shapes(
        &[mixed()],
        &ExtrudeOptions {
            depth: 0.7,
            steps: 2,
            bevel_enabled: false,
            curve_segments: 3,
            ..Default::default()
        },
    );
    assert_digest(
        "extrude_nobevel",
        &no_bevel,
        f.get("extrude_nobevel").unwrap(),
    );

    let bevel = ExtrudeGeometry::from_shapes(
        &[mixed(), triangle()],
        &ExtrudeOptions {
            curve_segments: 3,
            ..Default::default()
        },
    );
    assert_digest("extrude_bevel", &bevel, f.get("extrude_bevel").unwrap());

    let opts = ExtrudeGeometry::from_shapes(
        &[triangle()],
        &ExtrudeOptions {
            depth: 0.35,
            bevel_enabled: true,
            bevel_size: Some(0.08),
            bevel_thickness: 0.08,
            bevel_offset: 0.01,
            bevel_segments: 2,
            ..Default::default()
        },
    );
    assert_geometry(
        "extrude_bevel_opts",
        &opts,
        f.get("extrude_bevel_opts").unwrap(),
    );
}

#[test]
fn extrude_along_path_matches_three() {
    let f = fixture();
    let to3 = |a: Vec<f64>| {
        a.chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect::<Vec<_>>()
    };
    let path = ExtrudePath {
        points: to3(nums(f.get("path_points").unwrap())),
        tangents: to3(nums(f.get("path_tangents").unwrap())),
    };
    let g = ExtrudeGeometry::from_shapes(
        &[triangle()],
        &ExtrudeOptions {
            steps: 6,
            extrude_path: Some(path),
            ..Default::default()
        },
    );
    assert_geometry("extrude_path", &g, f.get("extrude_path").unwrap());
}

#[test]
fn font_shapes_match_three() {
    let f = fixture();
    let font = Font::parse(FONT).expect("font");
    let shapes = font.generate_shapes("Ag\no#", 10.0);
    let want = f.get("text_shapes").unwrap().as_array().unwrap();
    assert_eq!(shapes.len(), want.len(), "shape count");
    for (s, w) in shapes.iter().zip(want) {
        assert_eq!(flat(&s.get_points(3)), nums(w.get("outline").unwrap()));
        let holes = w.get("holes").unwrap().as_array().unwrap();
        assert_eq!(s.holes.len(), holes.len(), "hole count");
        for (h, wh) in s.holes.iter().zip(holes) {
            assert_eq!(flat(&h.get_points(3)), nums(wh));
        }
    }
}

#[test]
fn text_geometry_matches_three() {
    let f = fixture();
    let font = Font::parse(FONT).expect("font");
    // `é` is not in the subset: it falls back to `?`, as in three.js.
    let text = TextGeometry::from_font(
        &font,
        "Agé",
        &TextOptions {
            size: 1.0,
            depth: Some(0.2),
            curve_segments: 3,
            bevel_enabled: true,
            bevel_thickness: 0.03,
            bevel_size: 0.02,
            bevel_segments: 2,
            ..Default::default()
        },
    );
    assert_digest("text", &text, f.get("text").unwrap());

    let legacy_height = TextGeometry::from_font(
        &font,
        "o",
        &TextOptions {
            size: 2.0,
            height: Some(0.5),
            curve_segments: 2,
            ..Default::default()
        },
    );
    assert_geometry(
        "text_default",
        &legacy_height,
        f.get("text_default").unwrap(),
    );
}
