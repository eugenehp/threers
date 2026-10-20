//! Generators against three.js r165's own buffers.
//!
//! `testdata/three_r165_generators.json` was written by node running
//! `three@0.165.0` (web/node_modules/three): each entry holds the inputs and
//! the attribute / index / group arrays three.js produced, `f32` values in
//! their shortest round-tripping decimal. Sampled curve points and matrices
//! are full `f64`, as JavaScript held them.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::loaders::json::{parse, Value};

use super::decal_geometry::EulerOrder;
use super::{
    CapsuleGeometry, ConvexGeometry, DecalGeometry, EdgesGeometry, LatheGeometry,
    RoundedBoxGeometry, TeapotGeometry, TeapotOptions, TubeGeometry, WireframeGeometry,
};

fn fixtures() -> Value {
    parse(include_str!("testdata/three_r165_generators.json")).expect("fixture JSON")
}

fn nums(v: &Value, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixture key {key}"))
        .iter()
        .map(|x| x.as_number().unwrap())
        .collect()
}

fn triples(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Rebuild a fixture's source geometry (attributes, index, groups).
fn source(v: &Value) -> BufferGeometry {
    let mut g = BufferGeometry::new();
    for (name, size) in [("position", 3), ("normal", 3), ("uv", 2)] {
        if v.get(name).is_some() {
            let a: Vec<f32> = nums(v, name).iter().map(|&x| x as f32).collect();
            g.set_attribute(name, BufferAttribute::new(a, size));
        }
    }
    if v.get("index").is_some() {
        g.set_index(nums(v, "index").iter().map(|&x| x as u32).collect());
    }
    if v.get("groups").is_some() {
        for c in nums(v, "groups").chunks_exact(3) {
            g.add_group(c[0] as u32, c[1] as u32, c[2] as u32);
        }
    }
    g
}

/// Compare `got` with three.js's buffers: every attribute three.js wrote,
/// within `ulps` `f32` units in the last place (0 = bit for bit) or 1e-12
/// absolute, and index and
/// groups exactly.
fn check(name: &str, got: &BufferGeometry, want: &Value, ulps: u32) {
    for attr in ["position", "normal", "uv"] {
        if want.get(attr).is_none() {
            continue;
        }
        let w = nums(want, attr);
        let g = &got
            .get_attribute(attr)
            .unwrap_or_else(|| panic!("{name}: no {attr}"))
            .array;
        assert_eq!(g.len(), w.len(), "{name}.{attr} length");
        let mut worst = (0u32, 0usize);
        for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
            let b = b as f32;
            let d = if a == b {
                0
            } else if a.is_sign_negative() == b.is_sign_negative() {
                (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
            } else {
                // Across zero: count the steps on both sides.
                a.to_bits().wrapping_sub(0x8000_0000).min(a.to_bits())
                    + b.to_bits().wrapping_sub(0x8000_0000).min(b.to_bits())
            };
            // Below 1e-12 a difference is a libm's last-bit choice in
            // `f64` showing through at zero, not a different value.
            let d = if (a as f64 - b as f64).abs() <= 1e-12 { 0 } else { d };
            if d > worst.0 {
                worst = (d, i);
            }
        }
        assert!(
            worst.0 <= ulps,
            "{name}.{attr}[{}]: {} vs three.js r165 {} ({} ulps)",
            worst.1,
            g[worst.1],
            w[worst.1] as f32,
            worst.0
        );
    }
    if want.get("index").is_some() {
        let w: Vec<u32> = nums(want, "index").iter().map(|&x| x as u32).collect();
        assert_eq!(got.index.as_deref(), Some(&w[..]), "{name}.index");
    } else {
        assert!(got.index.is_none(), "{name}: three.js is unindexed");
    }
    let w: Vec<u32> = if want.get("groups").is_some() {
        nums(want, "groups").iter().map(|&x| x as u32).collect()
    } else {
        Vec::new()
    };
    let g: Vec<u32> = got
        .groups
        .iter()
        .flat_map(|g| [g.start, g.count, g.material_index])
        .collect();
    assert_eq!(g, w, "{name}.groups");
}

#[test]
fn capsule_matches_r165() {
    let f = fixtures();
    let c = f.get("capsule").unwrap();
    let a = nums(c, "args");
    let g = CapsuleGeometry::with_f64(a[0], a[1], a[2] as usize, a[3] as usize);
    // The poles are put exactly on the axis (three.js leaves them ~1e-17
    // off), so position gets an absolute allowance there, nothing else.
    check_with_poles("capsule", &g, c);
}

fn check_with_poles(name: &str, got: &BufferGeometry, want: &Value) {
    let w = nums(want, "position");
    let g = &got.get_attribute("position").unwrap().array;
    assert_eq!(g.len(), w.len());
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        let b = b as f32;
        assert!(a == b || (a - b).abs() < 1e-15, "{name}.position[{i}]: {a} vs {b}");
    }
    let mut rest = want.clone();
    if let Some(o) = rest.as_object_mut() {
        o.remove("position");
    }
    check(name, got, &rest, 0);
}

#[test]
fn lathe_matches_r165() {
    let f = fixtures();
    for key in ["lathe", "lathe_clamped"] {
        let c = f.get(key).unwrap();
        let pts: Vec<[f64; 2]> = nums(c, "points").chunks_exact(2).map(|p| [p[0], p[1]]).collect();
        let seg = c.get("segments").unwrap().as_number().unwrap() as usize;
        let ps = c.get("phiStart").unwrap().as_number().unwrap();
        let pl = c.get("phiLength").unwrap().as_number().unwrap();
        check(key, &LatheGeometry::with_f64(&pts, seg, ps, pl), c, 0);
    }
}

#[test]
fn tube_matches_r165() {
    let f = fixtures();
    for key in ["tube_open", "tube_closed"] {
        let c = f.get(key).unwrap();
        let g = TubeGeometry::from_samples(
            &triples(&nums(c, "points")),
            &triples(&nums(c, "tangents")),
            c.get("radius").unwrap().as_number().unwrap(),
            c.get("radial").unwrap().as_number().unwrap() as usize,
            c.get("closed").unwrap().as_bool().unwrap(),
        );
        check(key, &g, c, 0);
    }
}

#[test]
fn convex_matches_r165() {
    let f = fixtures();
    let c = f.get("convex").unwrap();
    check("convex", &ConvexGeometry::from_f64(&triples(&nums(c, "points"))), c, 0);
}

#[test]
fn decal_matches_r165() {
    let f = fixtures();
    let c = f.get("decal").unwrap();
    let src = c.get("source").unwrap();
    let pos: Vec<f32> = nums(src, "position").iter().map(|&x| x as f32).collect();
    let nrm: Vec<f32> = nums(src, "normal").iter().map(|&x| x as f32).collect();
    let idx: Vec<u32> = nums(src, "index").iter().map(|&x| x as u32).collect();
    let mw: [f64; 16] = nums(c, "matrixWorld").try_into().unwrap();
    let p = nums(c, "decalPosition");
    let o = nums(c, "orientation");
    let s = nums(c, "size");
    let g = DecalGeometry::project(
        &pos,
        &nrm,
        Some(&idx),
        &mw,
        [p[0], p[1], p[2]],
        [o[0], o[1], o[2]],
        EulerOrder::parse(c.get("order").unwrap().as_str().unwrap()),
        [s[0], s[1], s[2]],
    );
    check("decal", &g, c, 0);
}

#[test]
fn rounded_box_matches_r165() {
    let f = fixtures();
    for key in ["rounded_box", "rounded_box_flat"] {
        let c = f.get(key).unwrap();
        let a = nums(c, "args");
        // `acos` in `angleTo`: the platform libm and V8's can differ in the
        // last `f64` bit, which flips the `f32` rounding of a rare uv.
        check(key, &RoundedBoxGeometry::new(a[0], a[1], a[2], a[3] as usize, a[4]), c, 1);
    }
}

#[test]
fn teapot_matches_r165() {
    let f = fixtures();
    for key in ["teapot", "teapot_opts"] {
        let c = f.get(key).unwrap();
        let a = c.get("args").unwrap().as_array().unwrap();
        let b = |i: usize| a[i].as_bool().unwrap();
        let o = TeapotOptions {
            size: a[0].as_number().unwrap(),
            segments: a[1].as_number().unwrap() as usize,
            bottom: b(2),
            lid: b(3),
            body: b(4),
            fit_lid: b(5),
            blinn: b(6),
        };
        check(key, &TeapotGeometry::new(o), c, 0);
    }
}

#[test]
fn edges_match_r165() {
    let f = fixtures();
    for key in ["edges_box", "edges_sphere"] {
        let c = f.get(key).unwrap();
        let src = source(c.get("source").unwrap());
        let t = c.get("threshold").unwrap().as_number().unwrap();
        check(key, &EdgesGeometry::with_threshold_degrees(&src, t), c, 0);
    }
}

#[test]
fn wireframe_matches_r165() {
    let f = fixtures();
    for key in ["wire_box", "wire_groups", "wire_sphere"] {
        let c = f.get(key).unwrap();
        let src = source(c.get("source").unwrap());
        check(key, &WireframeGeometry::new(&src), c, 0);
    }
}
