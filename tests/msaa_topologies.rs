//! Skinned meshes, instanced meshes and opaque shader materials keep MSAA:
//! a frame holding one is multisampled — it used to fall back to one sample
//! for the whole frame — and draws what a single-sampled frame does, but
//! for its edges.
//!
//! Skips without a GPU adapter.

use threers::core::{Bone, BufferAttribute, InstancedMesh, Skeleton, SkinnedMesh};
use threers::prelude::*;

const N: u32 = 96;

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

fn scene_with(build: &dyn Fn(&mut Scene)) -> Scene {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s.add_light(AmbientLight::new(Color::WHITE, 1.0));
    build(&mut s);
    s
}

#[test]
fn skinned_instanced_and_shader_draws_stay_multisampled() {
    let Ok(mut single) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut msaa = HeadlessRenderer::builder().size(N, N).build().unwrap();
    msaa.set_msaa(4);
    let red = || Material::Basic(BasicMaterial::new(Color::new(1.0, 0.2, 0.1)));
    type Build = Box<dyn Fn(&mut Scene)>;
    let cases: [(&str, Build); 3] = [
        (
            "skinned",
            Box::new(move |s: &mut Scene| {
                let bone = s.add(Object3D::group());
                let mut g = SphereGeometry::new(1.0, 24, 12);
                let n = g.get_attribute("position").unwrap().count();
                g.set_attribute("joint", BufferAttribute::new(vec![0.0; n * 4], 4));
                g.set_attribute(
                    "weight",
                    BufferAttribute::new((0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
                );
                s.add(Object3D::skinned_mesh(SkinnedMesh::new(
                    g,
                    red(),
                    Skeleton::new(vec![Bone {
                        node: bone,
                        inverse_bind: Matrix4::identity(),
                    }]),
                )));
            }),
        ),
        (
            "instanced",
            Box::new(move |s: &mut Scene| {
                let mut im = InstancedMesh::new(BoxGeometry::new(0.8, 0.8, 0.8), red(), 2);
                im.transforms[0] = Matrix4::translation(Vector3::new(-0.8, 0.0, 0.0));
                im.transforms[1] = Matrix4::translation(Vector3::new(0.8, 0.3, 0.0));
                s.add(Object3D::instanced_mesh(im));
            }),
        ),
        (
            "shader",
            Box::new(move |s: &mut Scene| {
                let m = ShaderMaterial::new(
                    "@fragment fn fs_main(in : VsOut) -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.9, 0.3, 1.0); }",
                );
                s.add(Object3D::mesh(Mesh::new(
                    SphereGeometry::new(1.0, 24, 12),
                    Material::Shader(m),
                )));
            }),
        ),
    ];
    for (what, build) in &cases {
        let a = single.render_to_rgba(&mut scene_with(build.as_ref()), &camera());
        let m = msaa.render_to_rgba(&mut scene_with(build.as_ref()), &camera());
        assert!(
            msaa.renderer().last_frame_multisampled(),
            "{what}: the frame fell back to one sample"
        );
        // The same picture, but where an edge crosses a pixel.
        let differ = a
            .chunks(4)
            .zip(m.chunks(4))
            .filter(|(p, q)| (0..3).any(|c| (p[c] as i32 - q[c] as i32).abs() > 8))
            .count();
        let drawn = a
            .chunks(4)
            .filter(|p| p[0] as u32 + p[1] as u32 > 40)
            .count();
        eprintln!("{what}: {differ} pixels differ (of {drawn} drawn)");
        assert!(drawn > 500, "{what}: nothing drawn");
        assert!(differ > 0, "{what}: multisampling changed no edge");
        assert!(
            differ < drawn / 3,
            "{what}: {differ} of {drawn} pixels differ"
        );
    }
}
