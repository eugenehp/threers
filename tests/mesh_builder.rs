//! `MeshBuilder` and `MeshReport`: what goes in, and that it faces the right way.

use threers::prelude::*;

fn normal_at(g: &BufferGeometry, i: usize) -> Vector3 {
    let n = &g.get_attribute("normal").unwrap().array;
    Vector3::new(n[i * 3], n[i * 3 + 1], n[i * 3 + 2])
}

/// The way a triangle actually faces, from its winding.
fn face(g: &BufferGeometry, t: usize) -> Vector3 {
    let p: Vec<Vector3> = g.positions().unwrap().collect();
    let [a, b, c] = g.triangle(t).map(|i| p[i as usize]);
    (b - a).cross(c - a).normalize()
}

fn square(order: [usize; 4]) -> MeshBuilder {
    let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut m = MeshBuilder::new();
    let ids: Vec<u32> = order
        .iter()
        .map(|&i| m.vertex(Vector3::new(corners[i].0, 0.0, corners[i].1)))
        .collect();
    m.facing(Vector3::UP).quad(ids[0], ids[1], ids[2], ids[3]);
    m
}

#[test]
fn facing_decides_the_side_whatever_the_order() {
    for order in [[0, 1, 2, 3], [3, 2, 1, 0]] {
        let g = square(order).build();
        for t in 0..g.triangle_count() {
            assert!(face(&g, t).y > 0.99, "{order:?}: {:?}", face(&g, t));
        }
        assert!(normal_at(&g, 0).y > 0.99);
    }
}

#[test]
fn a_ribbon_faces_up_with_its_edges_either_way_round() {
    let a: Vec<Vector3> = (0..5).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();
    let b: Vec<Vector3> = (0..5).map(|i| Vector3::new(i as f32, 0.0, 1.0)).collect();
    for (left, right) in [(&a, &b), (&b, &a)] {
        let mut m = MeshBuilder::new();
        m.ribbon(left, right, Vector3::UP, false);
        let g = m.build();
        assert_eq!(g.triangle_count(), 8);
        for t in 0..8 {
            assert!(face(&g, t).y > 0.99);
        }
    }
}

#[test]
fn a_closed_ribbon_has_no_ends() {
    let ring = |r: f32| -> Vec<Vector3> {
        (0..16)
            .map(|i| {
                let a = i as f32 / 16.0 * std::f32::consts::TAU;
                Vector3::new(a.cos() * r, 0.0, a.sin() * r)
            })
            .collect()
    };
    let mut m = MeshBuilder::new();
    m.ribbon(&ring(2.0), &ring(1.0), Vector3::UP, true);
    let report = MeshReport::of(&m.build());
    assert_eq!(report.flipped_edges, 0);
    // Only the two rims are open: the join has closed the loop.
    assert_eq!(report.open_edges, 32);
}

#[test]
fn a_grid_follows_its_height_and_slopes_shade() {
    let mut m = MeshBuilder::new();
    m.grid(10.0, 10.0, 10, 10, |x, _| x);
    let g = m.build();
    assert_eq!(g.triangle_count(), 200);
    let bounds = g.bounding_box.clone().unwrap();
    assert!((bounds.max.y - 5.0).abs() < 1e-4);
    // Rising towards +x, the surface leans towards -x.
    let n = normal_at(&g, 60);
    assert!(n.x < -0.6 && n.y > 0.6, "{n:?}");
    assert!(face(&g, 0).y > 0.0);
}

#[test]
fn appended_shapes_stay_solid_and_land_where_they_were_put() {
    let mut m = MeshBuilder::new();
    let cube = BoxGeometry::new(1.0, 1.0, 1.0);
    m.append(&cube, &Matrix4::translation(Vector3::new(10.0, 0.0, 0.0)));
    m.append(&cube, &Matrix4::translation(Vector3::new(-10.0, 0.0, 0.0)));
    let g = m.build();
    assert_eq!(g.triangle_count(), cube.triangle_count() * 2);
    let bounds = g.bounding_box.clone().unwrap();
    assert!((bounds.max.x - 10.5).abs() < 1e-4 && (bounds.min.x + 10.5).abs() < 1e-4);
    assert!(MeshReport::of(&g).is_solid(), "{:?}", MeshReport::of(&g));
}

/// Every triangle's winding must agree with the normals it is lit by.
#[test]
fn a_mirrored_shape_is_not_turned_inside_out() {
    let mirror = Matrix4::compose(
        Vector3::ZERO,
        Quaternion::IDENTITY,
        Vector3::new(-1.0, 1.0, 1.0),
    );
    let mut m = MeshBuilder::new();
    m.append(&BoxGeometry::new(1.0, 1.0, 1.0), &mirror);
    let g = m.build();
    for t in 0..g.triangle_count() {
        let [a, _, _] = g.triangle(t);
        assert!(
            face(&g, t).dot(normal_at(&g, a as usize)) > 0.9,
            "triangle {t}"
        );
    }
}

#[test]
fn normals_survive_stretching() {
    let stretch = Matrix4::compose(
        Vector3::ZERO,
        Quaternion::IDENTITY,
        Vector3::new(1.0, 10.0, 1.0),
    );
    let mut m = MeshBuilder::new();
    m.append(&SphereGeometry::new(1.0, 16, 12), &stretch);
    let g = m.build();
    // On a tall ellipsoid, surfaces at the waist face almost level.
    let p: Vec<Vector3> = g.positions().unwrap().collect();
    let waist = p
        .iter()
        .position(|v| v.y.abs() < 3.0 && v.y.abs() > 2.0)
        .unwrap();
    assert!(
        normal_at(&g, waist).y.abs() < 0.2,
        "{:?}",
        normal_at(&g, waist)
    );
}

#[test]
fn flat_shading_gives_each_face_its_own_normal() {
    let mut m = MeshBuilder::new();
    m.append(&BoxGeometry::new(1.0, 1.0, 1.0), &Matrix4::identity());
    let g = m.build_flat();
    assert_eq!(
        g.get_attribute("position").unwrap().count(),
        g.triangle_count() * 3
    );
    for t in 0..g.triangle_count() {
        let [a, _, _] = g.triangle(t);
        assert!(face(&g, t).dot(normal_at(&g, a as usize)) > 0.999);
    }
}

#[test]
fn colours_appear_only_when_used() {
    assert!(square([0, 1, 2, 3])
        .build()
        .get_attribute("color")
        .is_none());
    let mut m = MeshBuilder::new();
    m.color(Color::new(1.0, 0.0, 0.0));
    let a = m.vertex(Vector3::ZERO);
    m.color(Color::new(0.0, 1.0, 0.0));
    let b = m.vertex(Vector3::new(1.0, 0.0, 0.0));
    let c = m.vertex(Vector3::new(0.0, 0.0, 1.0));
    m.triangle(a, b, c);
    let g = m.build();
    assert_eq!(
        &g.get_attribute("color").unwrap().array[..6],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "vertex 3 does not exist")]
fn a_bad_index_is_caught_where_it_is_written() {
    let mut m = MeshBuilder::new();
    let a = m.vertex(Vector3::ZERO);
    m.triangle(a, a, 3);
}

#[test]
fn the_report_knows_a_solid_from_a_broken_one() {
    let cube = BoxGeometry::new(1.0, 1.0, 1.0);
    assert!(
        MeshReport::of(&cube).is_solid(),
        "{:?}",
        MeshReport::of(&cube)
    );

    let mut hole = cube.clone();
    let mut index = hole.index.clone().unwrap();
    index.truncate(index.len() - 3);
    hole.set_index(index);
    let report = MeshReport::of(&hole);
    assert_eq!(report.open_edges, 3);
    assert!(!report.is_solid());

    let mut flipped = cube.clone();
    let mut index = flipped.index.clone().unwrap();
    index.swap(0, 1);
    index.swap(3, 4);
    flipped.set_index(index);
    assert!(MeshReport::of(&flipped).flipped_edges > 0);

    let mut m = MeshBuilder::new();
    let a = m.vertex(Vector3::ZERO);
    let b = m.vertex(Vector3::new(1.0, 0.0, 0.0));
    let c = m.vertex(Vector3::new(2.0, 0.0, 0.0));
    m.triangle(a, b, c);
    assert_eq!(MeshReport::of(&m.build()).degenerate, 1);
}

#[test]
fn a_builder_mesh_is_seen_by_rays() {
    let mut m = MeshBuilder::new();
    m.grid(20.0, 20.0, 4, 4, |_, _| 1.0);
    let mut scene = Scene::new();
    let id = scene.add_mesh(m.build(), StandardMaterial::new(Color::WHITE).into());
    scene.update_world();
    let caster = Raycaster::new(
        Vector3::new(0.0, 10.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        0.0,
        100.0,
    );
    let hits = caster.intersect_object(&scene.arena, id, false);
    assert!((hits[0].distance - 9.0).abs() < 1e-3);
}
