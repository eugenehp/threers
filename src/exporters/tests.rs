//! Byte comparison against three.js r165's own exporters.
//!
//! `testdata/` was written by running `OBJExporter`, `STLExporter` and
//! `PLYExporter` from `web/node_modules/three` (0.165.0) over a small scene — a
//! rotated, scaled group holding an indexed box and a point cloud, an
//! unindexed triangle soup mirrored by a negative scale, a line and line
//! segments — and dumping each object's buffers and `matrixWorld` beside the
//! output (`scene.json`, `ply-scene.json`). These tests rebuild the same
//! objects from the dump and require the same bytes.

use super::*;
use crate::core::BufferAttribute;
use crate::loaders::json::{self, Value};

fn load(name: &str) -> Vec<ExportObject> {
    let path = format!("{}/src/exporters/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let doc = json::parse(&text).unwrap();
    let floats = |v: &Value| -> Option<Vec<f32>> {
        v.as_array()
            .map(|a| a.iter().map(|x| x.as_number().unwrap() as f32).collect())
    };
    doc.get("objects")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            let mut g = BufferGeometry::new();
            for (name, size) in [("position", 3), ("normal", 3), ("uv", 2), ("color", 3)] {
                if let Some(a) = o.get(name).and_then(floats) {
                    g.set_attribute(name, BufferAttribute::new(a, size));
                }
            }
            if let Some(i) = o.get("index").and_then(|v| v.as_array()) {
                g.set_index(i.iter().map(|x| x.as_u64().unwrap() as u32).collect());
            }
            let m: Vec<f64> = o
                .get("matrixWorld")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_number().unwrap())
                .collect();
            let kind = match o.get("kind").unwrap().as_str().unwrap() {
                "mesh" => ExportKind::Mesh,
                "points" => ExportKind::Points,
                "line" => ExportKind::Line,
                "linesegments" => ExportKind::LineSegments,
                k => panic!("{k}"),
            };
            ExportObject {
                name: o.get("name").unwrap().as_str().unwrap().into(),
                kind,
                geometry: Arc::new(g),
                matrix_world: m.try_into().unwrap(),
                material_name: o.get("material").unwrap().as_str().unwrap().into(),
            }
        })
        .collect()
}

fn expected(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/src/exporters/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// Equal, or the first differing line for the failure message.
fn same_text(got: &str, want: &[u8], what: &str) {
    let want = std::str::from_utf8(want).unwrap();
    if got != want {
        for (i, (a, b)) in got.lines().zip(want.lines()).enumerate() {
            assert_eq!(a, b, "{what}: line {}", i + 1);
        }
        assert_eq!(got.len(), want.len(), "{what}: length");
    }
}

#[test]
fn obj_matches_three_js() {
    let objects = load("scene.json");
    same_text(
        &write_obj(&objects, &ObjOptions::default()),
        &expected("scene.obj"),
        "obj",
    );
}

#[test]
fn stl_matches_three_js() {
    let objects = load("scene.json");
    same_text(
        &write_stl_ascii(&objects, NumberFormat::JavaScript),
        &expected("scene.stl"),
        "ascii stl",
    );
    assert_eq!(write_stl_binary(&objects), expected("scene-bin.stl"), "binary stl");
}

#[test]
fn ply_matches_three_js() {
    let objects = load("ply-scene.json");
    let text = |o: &PlyOptions| String::from_utf8(write_ply(&objects, o).unwrap()).unwrap();
    same_text(&text(&PlyOptions::default()), &expected("scene.ply"), "ascii ply");
    same_text(
        &text(&PlyOptions {
            exclude_attributes: vec!["index".into(), "uv".into()],
            ..Default::default()
        }),
        &expected("scene-noidx.ply"),
        "ascii ply, no index or uv",
    );
    let binary = |little_endian| {
        write_ply(
            &objects,
            &PlyOptions {
                binary: true,
                little_endian,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(binary(false), expected("scene-be.ply"), "big-endian ply");
    assert_eq!(binary(true), expected("scene-le.ply"), "little-endian ply");
}

/// Points in the scene: three.js drops the face element entirely.
#[test]
fn ply_with_points_matches_three_js() {
    let objects = load("scene.json");
    let got = String::from_utf8(write_ply(&objects, &PlyOptions::default()).unwrap()).unwrap();
    same_text(&got, &expected("scene-points.ply"), "ply with points");
}

/// A strip of three quads, each two fanned triangles.
fn strip(grouped: bool) -> BufferGeometry {
    let mut g = BufferGeometry::new();
    let mut p = Vec::new();
    for y in 0..=1 {
        for x in 0..=3 {
            p.extend_from_slice(&[x as f32, y as f32, 0.0]);
        }
    }
    g.set_attribute("position", BufferAttribute::new(p, 3));
    let mut idx = Vec::new();
    for x in 0..3u32 {
        let [a, b, c, d] = [x, x + 1, x + 5, x + 4];
        idx.extend_from_slice(&[a, b, c, a, c, d]);
    }
    g.set_index(idx);
    if grouped {
        g.set_polygons(vec![2, 2, 2]);
    }
    g
}

#[test]
fn polygons_are_written_as_n_gons_when_asked() {
    let objects = [ExportObject::mesh("", strip(true))];
    let faces = |s: String| -> Vec<String> {
        s.lines()
            .filter(|l| l.starts_with("f "))
            .map(String::from)
            .collect()
    };
    assert_eq!(faces(write_obj(&objects, &ObjOptions::default())).len(), 6);
    assert_eq!(
        faces(write_obj(
            &objects,
            &ObjOptions {
                polygons: true,
                ..Default::default()
            }
        )),
        ["f 1 2 6 5", "f 2 3 7 6", "f 3 4 8 7"]
    );
    let ply = String::from_utf8(
        write_ply(
            &objects,
            &PlyOptions {
                polygons: true,
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .unwrap();
    assert!(ply.contains("element face 3\n"), "{ply}");
    assert!(ply.ends_with("\n4 0 1 5 4\n4 1 2 6 5\n4 2 3 7 6\n\n"), "{ply}");
    // No grouping: triangles whatever the option says.
    let plain = [ExportObject::mesh("", strip(false))];
    assert_eq!(
        faces(write_obj(
            &plain,
            &ObjOptions {
                polygons: true,
                ..Default::default()
            }
        ))
        .len(),
        6
    );
}

#[test]
fn a_fractional_face_count_is_refused_like_three_js() {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(vec![0.0; 12], 3));
    assert!(write_ply(&[ExportObject::mesh("", g)], &PlyOptions::default()).is_none());
}

#[test]
fn arena_traversal_composes_world_matrices() {
    use crate::core::{Mesh, Object3D, ObjectArena};
    let mut arena = ObjectArena::new();
    let mut parent = Object3D::group();
    parent.position = crate::math::Vector3::new(1.0, 0.0, 0.0);
    let parent = arena.insert(parent);
    let mut child = Object3D::mesh(Mesh::new(
        strip(false),
        crate::materials::BasicMaterial::default().into(),
    ));
    child.position = crate::math::Vector3::new(0.0, 2.0, 0.0);
    child.name = "c".into();
    let child = arena.insert(child);
    arena.add_child(parent, child);
    let objects = ExportObject::from_arena(&arena, &[parent]);
    assert_eq!(objects.len(), 1);
    let obj = write_obj(&objects, &ObjOptions::default());
    assert!(obj.starts_with("o c\nv 1 2 0\n"), "{obj}");
}
