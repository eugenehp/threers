//! Native mesh-bvh picking smoke test.
//!
//! ```text
//! cargo run --example mesh_bvh_picking --features mesh-bvh
//! ```

use threers::mesh_bvh::BuildOptions;
use threers::{
    BasicMaterial, Color, Mesh, MeshBvh, Object3D, ObjectArena, Ray, Raycaster, SphereGeometry,
    Vector3,
};

fn main() {
    let mut geom = SphereGeometry::new(1.0, 32, 16);
    let bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
    geom.compute_bounds_tree(BuildOptions::default());

    let direct_hits = bvh.raycast(
        &Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0)),
        0.0,
        f32::INFINITY,
        false,
    );

    let mut arena = ObjectArena::new();
    let root = arena.insert(Object3D::group());
    let mat = threers::Material::Basic(BasicMaterial::new(Color::from_hex(0x4488cc)));
    let mesh_id = arena.insert(Object3D::mesh(Mesh::new(geom, mat)));
    arena.add_child(root, mesh_id);
    arena.update_world_matrices(root, threers::Matrix4::identity());

    let rc = Raycaster::new(
        Vector3::new(0.0, 0.0, -5.0),
        Vector3::new(0.0, 0.0, 1.0),
        0.0,
        f32::INFINITY,
    );
    let scene_hits = rc.intersect_objects(&arena, root, true);

    assert!(!direct_hits.is_empty(), "BVH raycast should hit sphere");
    assert_eq!(
        direct_hits.len(),
        scene_hits.len(),
        "BVH and scene raycaster hit counts differ"
    );
    println!(
        "mesh_bvh_picking: {} hit(s), closest distance {:.4}",
        direct_hits.len(),
        direct_hits[0].distance
    );
}
