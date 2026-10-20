//! A blended DoubleSide mesh is drawn as three.js draws it: back faces, then
//! front faces (`WebGLRenderer.renderObject` with `forceSinglePass: false`).
//!
//! One mesh, two quads at half opacity: a red one facing the camera at
//! z = +0.5, listed FIRST, and a green one facing away at z = −0.5. Drawn
//! back-then-front, the green back face goes down first and the red front face
//! blends over it, so red dominates; in one pass in index order the green would
//! land on top. Skipped without a GPU adapter.

use std::sync::Arc;
use threers::cameras::OrthographicCamera;
use threers::core::{BufferAttribute, BufferGeometry, Mesh, Object3D};
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

const N: u32 = 32;

fn quads() -> BufferGeometry {
    let mut g = BufferGeometry::new();
    // near quad (z = +0.5, wound to face +z) then far quad (z = −0.5, wound to face −z)
    let p = [
        -1.0f32, -1.0, 0.5, 1.0, -1.0, 0.5, 1.0, 1.0, 0.5, -1.0, 1.0, 0.5, //
        -1.0, -1.0, -0.5, 1.0, -1.0, -0.5, 1.0, 1.0, -0.5, -1.0, 1.0, -0.5,
    ];
    let n = [0.0f32, 0.0, 1.0].repeat(4).into_iter().chain([0.0f32, 0.0, -1.0].repeat(4)).collect::<Vec<_>>();
    // red texel for the near quad, green for the far one
    let uv = [0.25f32, 0.5].repeat(4).into_iter().chain([0.75f32, 0.5].repeat(4)).collect::<Vec<_>>();
    g.set_attribute("position", BufferAttribute::new(p.to_vec(), 3));
    g.set_attribute("normal", BufferAttribute::new(n, 3));
    g.set_attribute("uv", BufferAttribute::new(uv, 2));
    g.index = Some(vec![0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6]);
    g
}

#[test]
fn a_double_sided_translucent_mesh_draws_its_back_faces_first() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut tex = Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![255, 0, 0, 255, 0, 255, 0, 255]);
    tex.mag_filter = TextureFilter::Nearest;
    tex.min_filter = TextureFilter::Nearest;
    let mut m = BasicMaterial::new(Color::WHITE);
    m.map = Some(Arc::new(tex));
    m.opacity = 0.5;
    m.transparent = true;
    m.side = 2;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(quads(), Material::from(m))));
    let mut cam = OrthographicCamera::new(-2.0, 2.0, 2.0, -2.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    let px = r.render_to_rgba(&mut scene, &cam);
    let c = (((N / 2) * N + N / 2) * 4) as usize;
    let (red, green) = (px[c], px[c + 1]);
    assert!(red > 0 && green > 0, "both quads should contribute: rgb {:?}", &px[c..c + 3]);
    assert!(red > green, "the back face blended over the front face: red {red}, green {green}");
}
