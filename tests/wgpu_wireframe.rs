//! `wireframe: true` on the native wgpu backend, checked on real pixels.
//!
//! A quad whose geometry records its polygons must draw its four edges and not
//! the diagonal that splits it into triangles — for a plain mesh, an instanced
//! mesh and a skinned mesh, each of which has its own pipeline — and a polygon
//! table set after the geometry is already on the GPU must take effect.
//!
//! The probes are shared with `tests/metal_backend.rs` (`tests/wire_probe`),
//! so the two backends are held to the same picture. Skipped when no GPU
//! adapter is available.

use std::sync::Arc;
use threers::cameras::OrthographicCamera;
use threers::core::{
    Bone, BufferGeometry, InstancedMesh, Mesh, Object3D, ObjectId, ObjectKind, Skeleton,
    SkinnedMesh,
};
use threers::geometries::QuadGeometry;
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Matrix4, Vector3};
use threers::scene::Scene;
use threers::HeadlessRenderer;

#[path = "wire_probe/mod.rs"]
mod wire_probe;
use wire_probe::{
    assert_outline_only, corners, diagonals_drawn, interior_lit, plane, wire_material as wire, SIZE,
};

fn renderer() -> Option<HeadlessRenderer> {
    let mut r = HeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    // Hardware lines, unblended: a lit pixel is a lit pixel.
    r.set_msaa(1);
    Some(r)
}

/// Looks down -Z at a 2x2 window around the origin.
fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

fn quad() -> BufferGeometry {
    let c = corners().map(|(x, y)| Vector3::new(x, y, 0.0));
    QuadGeometry::new(c)
}

fn scene_with(obj: Object3D) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(obj);
    scene
}

#[test]
fn wireframe_quads_draw_outline_without_diagonal() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Control: drop the table and it is three.js's per-triangle wireframe,
    // which draws one diagonal. Proves the probe can see a diagonal at all.
    let mut control = plane();
    control.polygons = None;
    let rgba = r.render_to_rgba(
        &mut scene_with(Object3D::mesh(Mesh::new(control, wire()))),
        &camera(),
    );
    assert_eq!(
        diagonals_drawn(&rgba),
        1,
        "triangle wireframe shows its diagonal"
    );

    for (what, g) in [("PlaneGeometry", plane()), ("QuadGeometry", quad())] {
        assert!(g.polygon_counts().is_some(), "{what} records its quad");
        let mut scene = scene_with(Object3D::mesh(Mesh::new(g, wire())));
        let rgba = r.render_to_rgba(&mut scene, &camera());
        assert_outline_only(&rgba, what);
    }
}

#[test]
fn instanced_wireframe_draws_outlines() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Filled first: the instanced pipeline itself draws here.
    let mut filled = InstancedMesh::new(
        plane(),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
        1,
    );
    filled.set_matrix_at(0, Matrix4::identity());
    let rgba = r.render_to_rgba(&mut scene_with(Object3D::instanced_mesh(filled)), &camera());
    assert!(interior_lit(&rgba), "filled instance draws its interior");

    // Two instances: one on the probe, one off to the side.
    let mut im = InstancedMesh::new(plane(), wire(), 2);
    im.set_matrix_at(0, Matrix4::identity());
    im.set_matrix_at(1, Matrix4::translation(Vector3::new(5.0, 0.0, 0.0)));
    let rgba = r.render_to_rgba(&mut scene_with(Object3D::instanced_mesh(im)), &camera());
    assert_outline_only(&rgba, "instanced");
}

#[test]
fn skinned_wireframe_draws_outlines() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let skeleton = || {
        Skeleton::new(vec![Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        }])
    };
    let filled = SkinnedMesh::from_unweighted(
        plane(),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
        skeleton(),
    );
    let rgba = r.render_to_rgba(&mut scene_with(Object3D::skinned_mesh(filled)), &camera());
    assert!(
        interior_lit(&rgba),
        "filled skinned mesh draws its interior"
    );

    let sm = SkinnedMesh::from_unweighted(plane(), wire(), skeleton());
    assert!(
        sm.geometry.polygon_counts().is_some(),
        "joint/weight writes keep the table"
    );
    let rgba = r.render_to_rgba(&mut scene_with(Object3D::skinned_mesh(sm)), &camera());
    assert_outline_only(&rgba, "skinned");
}

/// The GPU cache is keyed by the geometry's address and re-uploads only on a
/// version change, so a table attached to a geometry that is already uploaded
/// must bump the version or the old wire indices keep drawing.
#[test]
fn set_polygons_after_first_render_takes_effect() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut g = plane();
    g.polygons = None;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(Object3D::mesh(Mesh::new(g, wire())));
    let rgba = r.render_to_rgba(&mut scene, &camera());
    assert_eq!(diagonals_drawn(&rgba), 1, "untagged: the diagonal is drawn");

    let Some(ObjectKind::Mesh(mesh)) = scene.get_mut(id).map(|o| &mut o.kind) else {
        panic!("mesh object");
    };
    let before = Arc::as_ptr(&mesh.geometry);
    // Same allocation, so only the version can tell the cache it changed.
    Arc::get_mut(&mut mesh.geometry)
        .expect("sole owner")
        .set_polygons(vec![2]);
    assert_eq!(Arc::as_ptr(&mesh.geometry), before);

    let rgba = r.render_to_rgba(&mut scene, &camera());
    assert_outline_only(&rgba, "after set_polygons");
}
