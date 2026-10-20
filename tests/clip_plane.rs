//! `Object3D::clip_plane`, three.js's `material.clippingPlanes` for one plane:
//! a white quad over a blue background, clipped by the plane x ≥ 0, keeps its
//! right half and shows the background through its left — opaque and
//! blended, with and without MSAA. Skipped without a GPU adapter.

use threers::cameras::OrthographicCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::AmbientLight;
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Plane, Vector3};
use threers::scene::Scene;

const N: u32 = 64;

fn scene(transparent: bool, plane: Option<Plane>) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0000ff);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut m = BasicMaterial::new(Color::WHITE);
    if transparent {
        m.opacity = 0.999;
    }
    let mut obj = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), Material::from(m)));
    obj.clip_plane = plane;
    scene.add(obj);
    scene
}

fn red(rgba: &[u8], x: u32, y: u32) -> u8 {
    rgba[((y * N + x) * 4) as usize]
}

#[test]
fn a_clip_plane_discards_its_negative_side() {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    // keep x ≥ 0: normal +x, through the origin
    let keep_right = Plane::new(Vector3::new(1.0, 0.0, 0.0), 0.0);
    let (left, right, mid) = (N / 4, 3 * N / 4, N / 2);
    for msaa in [1, 4] {
        let Ok(mut r) = threers::HeadlessRenderer::builder().size(N, N).build() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        r.set_msaa(msaa);
        for transparent in [false, true] {
            let label = format!("msaa {msaa}, transparent {transparent}");
            let whole = r.render_to_rgba(&mut scene(transparent, None), &cam);
            assert!(red(&whole, left, mid) > 200 && red(&whole, right, mid) > 200, "{label}: unclipped quad not drawn");
            let cut = r.render_to_rgba(&mut scene(transparent, Some(keep_right)), &cam);
            assert!(red(&cut, left, mid) < 10, "{label}: the negative side was drawn");
            assert!(red(&cut, right, mid) > 200, "{label}: the positive side was clipped");
            // the flipped plane keeps the other half
            let flipped = Plane::new(Vector3::new(-1.0, 0.0, 0.0), 0.0);
            let other = r.render_to_rgba(&mut scene(transparent, Some(flipped)), &cam);
            assert!(red(&other, left, mid) > 200 && red(&other, right, mid) < 10, "{label}: the flipped plane");
        }
    }
}
