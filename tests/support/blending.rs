//! Alpha blending as three.js r165's default `WebGLRenderer` does it.
//!
//! With no render target three.js draws into the canvas's 8-bit framebuffer:
//! `colorspace_fragment` sRGB-encodes the shader output and `NormalBlending`
//! (`blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE,
//! ONE_MINUS_SRC_ALPHA)`) then mixes those *encoded* values with the encoded
//! bytes already there. (Only into a render target — an EffectComposer's
//! half-float buffer — is the output linear and the blend in linear light.)
//! So a 50% red quad over blue is (128, 0, 128), not the (188, 0, 188) a
//! linear blend gives. Shared by `tests/material_truth.rs` (wgpu) and
//! `tests/backend_parity.rs` (Metal) through `#[path]`.
#![allow(dead_code)]

use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::BasicMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;

pub const N: u32 = 32;

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

/// What draws the blended surface: three.js blends every kind of object the
/// same way — a sprite (`SpriteMaterial` is transparent by default), each
/// instance of an `InstancedMesh`, a `SkinnedMesh` in its pose — so each must
/// land on the same bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Front {
    Mesh,
    Sprite,
    Instanced,
    Skinned,
}

pub const FRONTS: [Front; 4] = [Front::Mesh, Front::Sprite, Front::Instanced, Front::Skinned];

/// `front` at `opacity` over an opaque `back`, both filling the frame.
pub fn scene(back: u32, front: u32, opacity: f32) -> Scene {
    scene_with(back, front, opacity, Front::Mesh)
}

/// The opaque back plane alone.
pub fn back_scene(back: u32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut b = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.5, 2.5),
        BasicMaterial::new(Color::from_hex(back)).into(),
    ));
    b.position = Vector3::new(0.0, 0.0, -0.5);
    scene.add(b);
    scene
}

/// [`scene`] with the blended surface drawn as `kind`.
pub fn scene_with(back: u32, front: u32, opacity: f32, kind: Front) -> Scene {
    use std::sync::Arc;
    use threers::core::{Bone, BufferAttribute, InstancedMesh, ObjectId, Skeleton, SkinnedMesh, Sprite};
    use threers::materials::{Material, SpriteMaterial};
    use threers::math::Matrix4;
    let mut scene = back_scene(back);
    let mut m = BasicMaterial::new(Color::from_hex(front));
    m.transparent = true;
    m.opacity = opacity;
    let mut f = match kind {
        Front::Mesh => Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), m.into())),
        Front::Sprite => {
            let mut sm = SpriteMaterial::new(Color::from_hex(front));
            sm.opacity = opacity;
            let mut o = Object3D::sprite(Sprite::new(Material::Sprite(sm)));
            o.scale = Vector3::new(2.5, 2.5, 1.0);
            o
        }
        Front::Instanced => {
            let mut im = InstancedMesh::from_arc(
                Arc::new(PlaneGeometry::new(2.5, 2.5)),
                Arc::new(m.into()),
                1,
            );
            im.set_matrix_at(0, Matrix4::identity());
            Object3D::instanced_mesh(im)
        }
        Front::Skinned => {
            let mut g = PlaneGeometry::new(2.5, 2.5);
            let n = g.get_attribute("position").unwrap().count();
            g.set_attribute("joint", BufferAttribute::new(vec![0.0; n * 4], 4));
            g.set_attribute("weight", BufferAttribute::new([1.0, 0.0, 0.0, 0.0].repeat(n), 4));
            let mut skeleton = Skeleton::new(vec![Bone {
                node: ObjectId::default(),
                inverse_bind: Matrix4::identity(),
            }]);
            // In front of the back plane whether the pose or the object
            // places it (z 0 or 0.5): only the blend is under test.
            skeleton.bone_matrices = vec![Matrix4::identity()];
            Object3D::skinned_mesh(SkinnedMesh::new(g, m.into(), skeleton))
        }
    };
    f.position = Vector3::new(0.0, 0.0, 0.5);
    scene.add(f);
    scene
}

/// three.js's bytes: `round(255 · (a · src + (1 − a) · dst))` per channel on
/// the encoded bytes.
pub fn expected(back: u32, front: u32, a: f32) -> [u8; 3] {
    let ch = |hex: u32, k: u32| ((hex >> (16 - 8 * k)) & 0xff) as f32 / 255.0;
    let mut out = [0u8; 3];
    for k in 0..3 {
        out[k as usize] = ((a * ch(front, k) + (1.0 - a) * ch(back, k)) * 255.0).round() as u8;
    }
    out
}

pub const CASES: [(u32, u32, f32); 3] = [
    (0x0000ff, 0xff0000, 0.5),
    (0x404040, 0x00ff00, 0.25),
    (0x2060a0, 0xffc040, 0.7),
];

/// Every case through `render`; returns the failures.
pub fn check(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    check_kinds(backend, &FRONTS, render)
}

/// [`check`] for the given kinds of blended surface.
pub fn check_kinds(
    backend: &str,
    kinds: &[Front],
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for &kind in kinds {
        for (back, front, a) in CASES {
            let rgba = render(&mut scene_with(back, front, a, kind), &cam);
            if let Some(f) = judge(backend, &format!("{kind:?}"), &rgba, back, front, a) {
                failures.push(f);
            }
        }
    }
    failures
}

/// The centre pixel of `rgba` against three.js's blend; a failure message
/// when it is more than a level off.
pub fn judge(backend: &str, what: &str, rgba: &[u8], back: u32, front: u32, a: f32) -> Option<String> {
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
    let want = expected(back, front, a);
    let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
    eprintln!("{backend} {what} {front:06x} @ {a} over {back:06x}: got {got:?} want {want:?}");
    (err > 1).then(|| {
        format!("{backend} {what}: {front:06x} at {a} over {back:06x} is {got:?}, three.js {want:?}")
    })
}
