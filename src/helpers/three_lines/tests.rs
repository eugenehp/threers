//! Each helper against three.js r165: the fixture holds the world-space line
//! points (and colours, and filled triangles) three.js's own helper objects
//! draw, flattened by `src/helpers/testdata/three_helpers_r165.json`'s
//! generator (the node script in the commit that added it): `Line` strips
//! expanded into segments, wireframe meshes into WebGL's wireframe edges.

use super::*;
use crate::loaders::json::{parse, Value};

fn fixture() -> Value {
    parse(include_str!("../testdata/three_helpers_r165.json")).expect("fixture parses")
}

fn nums(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(|a| a.as_array())
        .unwrap_or_else(|| panic!("{key} missing"))
        .iter()
        .map(|n| n.as_number().unwrap())
        .collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(|n| n.as_number()).unwrap()
}

fn mat(v: &Value, key: &str) -> Mat4 {
    nums(v, key).try_into().unwrap()
}

fn v3(v: &Value, key: &str) -> [f64; 3] {
    nums(v, key).try_into().unwrap()
}

fn c3(v: &Value, key: &str) -> [f32; 3] {
    let n = nums(v, key);
    [n[0] as f32, n[1] as f32, n[2] as f32]
}

/// Same length, and every value within `1e-5` relative (three.js's figures
/// went through `toPrecision(9)`; ours through `f32`).
fn assert_close(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 * w.abs().max(1.0);
        assert!(
            (g as f64 - w).abs() <= tol,
            "{name}[{i}]: got {g}, three.js {w}"
        );
    }
}

fn check(name: &str, case: &Value, got: &HelperGeometry) {
    assert_close(&format!("{name}.lines"), &got.lines, &nums(case, "lines"));
    assert_close(&format!("{name}.colors"), &got.colors, &nums(case, "colors"));
    assert_close(&format!("{name}.mesh"), &got.mesh, &nums(case, "mesh"));
}

#[test]
fn directional_light_matches_three() {
    let f = fixture();
    let c = f.get("directional").unwrap();
    check("directional", c, &directional_light(&mat(c, "light"), v3(c, "target"), num(c, "size")));
}

#[test]
fn hemisphere_light_matches_three() {
    let f = fixture();
    let c = f.get("hemisphere").unwrap();
    check(
        "hemisphere",
        c,
        &hemisphere_light(&mat(c, "light"), num(c, "size"), c3(c, "sky"), c3(c, "ground")),
    );
}

#[test]
fn point_light_matches_three() {
    let f = fixture();
    let c = f.get("point").unwrap();
    check("point", c, &point_light(&mat(c, "light"), num(c, "size")));
}

#[test]
fn spot_light_matches_three() {
    let f = fixture();
    let c = f.get("spot").unwrap();
    check(
        "spot",
        c,
        &spot_light(&mat(c, "light"), v3(c, "target"), num(c, "distance"), num(c, "angle")),
    );
}

#[test]
fn camera_matches_three() {
    let f = fixture();
    let c = f.get("camera").unwrap();
    // The projection is rebuilt from the camera's parameters, as the browser
    // side does, rather than read from three.js.
    let proj = perspective_projection(num(c, "fov"), num(c, "aspect"), num(c, "near"), num(c, "far"), 1.0, 0.0, 35.0, None);
    let inv = invert(&proj);
    let want = nums(c, "projInv");
    for (i, (&g, &w)) in inv.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-7 * w.abs().max(1.0), "projInv[{i}]: {g} vs {w}");
    }
    // three.js's colours are its working-space (linear) `Color`s.
    check("camera", c, &camera(&inv, &mat(c, "world"), camera_default_colors()));
}

#[test]
fn orthographic_projection_matches_three() {
    let f = fixture();
    let c = f.get("ortho").unwrap();
    let got = orthographic_projection(-4.0, 4.0, 3.0, -3.0, 0.5, 20.0, 2.0, None);
    let want = nums(c, "proj");
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-7 * w.abs().max(1.0), "proj[{i}]: {g} vs {w}");
    }
}

#[test]
fn arrow_matches_three() {
    let f = fixture();
    let c = f.get("arrow").unwrap();
    let q = arrow_quaternion(v3(c, "dir"));
    let world = compose(v3(c, "origin"), q, [1.0; 3]);
    check(
        "arrow",
        c,
        &arrow(&world, num(c, "length"), num(c, "headLength"), num(c, "headWidth")),
    );
}

#[test]
fn plane_matches_three() {
    let f = fixture();
    let c = f.get("plane").unwrap();
    check("plane", c, &plane(v3(c, "normal"), num(c, "constant"), num(c, "size")));
}

#[test]
fn skeleton_matches_three() {
    let f = fixture();
    let c = f.get("skeleton").unwrap();
    let bones: Vec<Mat4> = c
        .get("bones")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            b.as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_number().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let parents: Vec<i32> = nums(c, "parents").iter().map(|&p| p as i32).collect();
    check("skeleton", c, &skeleton(&mat(c, "root"), &bones, &parents));
}

#[test]
fn box_helpers_match_three() {
    let f = fixture();
    let c = f.get("box").unwrap();
    check("box", c, &box_lines(v3(c, "min"), v3(c, "max")));
    let c = f.get("box3").unwrap();
    check("box3", c, &box3(v3(c, "min"), v3(c, "max")));
    assert_eq!(box_lines([1.0; 3], [-1.0; 3]), HelperGeometry::default());
}

#[test]
fn vertex_normals_match_three() {
    let f = fixture();
    let c = f.get("normals").unwrap();
    let p: Vec<f32> = nums(c, "positions").iter().map(|&v| v as f32).collect();
    let n: Vec<f32> = nums(c, "normals").iter().map(|&v| v as f32).collect();
    check("normals", c, &vertex_normals(&mat(c, "world"), &p, &n, num(c, "size")));
}

#[test]
fn rect_area_light_matches_three() {
    let f = fixture();
    let c = f.get("rect").unwrap();
    check("rect", c, &rect_area_light(&mat(c, "light"), num(c, "width"), num(c, "height")));
    let got = rect_area_light_color(c3(c, "lightColor"), num(c, "intensity") as f32);
    let want = nums(c, "color");
    for k in 0..3 {
        assert!((got[k] as f64 - want[k]).abs() < 1e-6, "color[{k}]");
    }
}

#[test]
fn object_bounds_transforms_each_box() {
    // A unit cube moved to x = 2 and a point at the origin: the union.
    let mut moved = IDENTITY;
    moved[12] = 2.0;
    let cube: Vec<f32> = [-0.5f32, 0.5].iter().flat_map(|&x| [x, x, x]).collect();
    let (min, max) = object_bounds(&[moved, IDENTITY], &[&cube, &[0.0, 0.0, 0.0]]).unwrap();
    assert_eq!(min, [0.0, -0.5, -0.5]);
    assert_eq!(max, [2.5, 0.5, 0.5]);
    assert_eq!(object_bounds(&[IDENTITY], &[&[]]), None);
}
