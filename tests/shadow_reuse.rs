//! Shadow maps are redrawn when what they show changed, not every frame.
//!
//! With `shadowMap.autoUpdate` on (the default, as in three.js) the renderer
//! used to draw every caster into every shadow map on every render. They
//! depend on the casters and the lights, never on the camera, so a still
//! scene under a moving camera redrew the same maps each frame. Now a render
//! whose casters and lights match the last one keeps its maps — and anything
//! that would change them still redraws them.
//!
//! Skipped when no GPU adapter is available.

use std::sync::Arc;

use threers::lights::Light;
use threers::prelude::*;

const W: u32 = 128;

fn scene() -> (Scene, threers::core::ObjectId, threers::core::ObjectId) {
    let mut s = Scene::new();
    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.5, -1.0, -0.3).normalize();
    sun.cast_shadow = true;
    let light = s.add(Object3D::light(Light::Directional(sun)));
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xcc6633))),
    ));
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    cube.cast_shadow = true;
    let cube = s.add(cube);
    (s, cube, light)
}

fn camera(x: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 50.0);
    c.position = Vector3::new(x, 4.0, 6.0);
    c.look_at(Vector3::ZERO);
    c
}

#[test]
fn shadow_maps_are_redrawn_only_when_they_would_change() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let (mut scene, cube, light) = scene();
    let updates = |hr: &mut HeadlessRenderer| hr.renderer().shadow_map_updates();

    hr.render(&mut scene, &camera(0.0));
    let first = updates(&mut hr);
    assert_eq!(first, 1, "the first render draws them");

    // The camera moves; nothing the maps show does.
    for x in [0.5, 1.0, 1.5] {
        hr.render(&mut scene, &camera(x));
    }
    assert_eq!(updates(&mut hr), first, "a moving camera reuses them");
    let reused = hr.read_rgba();

    // A caster moves: redrawn.
    scene.get_mut(cube).unwrap().position.x = 1.0;
    hr.render(&mut scene, &camera(1.5));
    assert_eq!(updates(&mut hr), first + 1, "a moved caster redraws them");

    // Its geometry is replaced in place, same pointer, new version: redrawn.
    if let ObjectKind::Mesh(m) = &mut scene.get_mut(cube).unwrap().kind {
        let g = Arc::make_mut(&mut m.geometry);
        g.bump_version();
    }
    hr.render(&mut scene, &camera(1.5));
    assert_eq!(updates(&mut hr), first + 2, "edited geometry redraws them");

    // The light turns: redrawn.
    if let ObjectKind::Light(Light::Directional(l)) = &mut scene.get_mut(light).unwrap().kind {
        l.direction = Vector3::new(0.5, -1.0, -0.3).normalize();
    }
    hr.render(&mut scene, &camera(1.5));
    assert_eq!(updates(&mut hr), first + 3, "a turned light redraws them");

    // And reused maps draw what redrawn ones would: the same scene from a
    // fresh renderer, which has to draw them.
    let (mut fresh_scene, _, _) = scene_at_reuse();
    let Ok(mut fresh) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let redrawn = fresh.render_to_rgba(&mut fresh_scene, &camera(1.5));
    assert_eq!(reused, redrawn, "reused maps change nothing");
}

/// The scene as it stood when the maps were being reused.
fn scene_at_reuse() -> (Scene, threers::core::ObjectId, threers::core::ObjectId) {
    scene()
}

#[test]
fn instanced_casters_reuse_until_an_instance_moves() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let (mut scene, cube, _) = scene();
    scene.get_mut(cube).unwrap().cast_shadow = false;
    let mut im = InstancedMesh::new(
        BoxGeometry::new(0.5, 0.5, 0.5),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
        3,
    );
    for i in 0..3 {
        im.set_matrix_at(
            i,
            Matrix4::translation(Vector3::new(i as f32 - 1.0, 0.25, 0.0)),
        );
    }
    let mut obj = Object3D::instanced_mesh(im);
    obj.cast_shadow = true;
    let id = scene.add(obj);
    let updates = |hr: &mut HeadlessRenderer| hr.renderer().shadow_map_updates();

    hr.render(&mut scene, &camera(0.0));
    let first = updates(&mut hr);
    for x in [0.5, 1.0] {
        hr.render(&mut scene, &camera(x));
    }
    assert_eq!(updates(&mut hr), first, "still instances reuse the maps");

    if let ObjectKind::InstancedMesh(im) = &mut scene.get_mut(id).unwrap().kind {
        im.set_matrix_at(1, Matrix4::translation(Vector3::new(0.0, 1.5, 0.0)));
    }
    hr.render(&mut scene, &camera(1.0));
    assert_eq!(updates(&mut hr), first + 1, "a moved instance redraws them");
    hr.render(&mut scene, &camera(1.5));
    assert_eq!(
        updates(&mut hr),
        first + 1,
        "and then they are reused again"
    );
}

#[test]
fn skinned_casters_reuse_until_the_pose_changes() {
    use threers::core::{Bone, BufferAttribute, Skeleton, SkinnedMesh};
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let (mut scene, cube, _) = scene();
    scene.get_mut(cube).unwrap().cast_shadow = false;
    let mut g = BoxGeometry::new(1.0, 1.0, 1.0);
    let count = g.get_attribute("position").unwrap().count();
    let joints = vec![0.0f32; count * 4];
    let weights: Vec<f32> = (0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
    g.set_attribute("joint", BufferAttribute::new(joints, 4));
    g.set_attribute("weight", BufferAttribute::new(weights, 4));
    let mut skeleton = Skeleton::new(vec![Bone {
        node: threers::core::ObjectId::default(),
        inverse_bind: Matrix4::identity(),
    }]);
    skeleton.bone_matrices = vec![Matrix4::translation(Vector3::new(0.0, 0.5, 0.0))];
    let mut obj = Object3D::skinned_mesh(SkinnedMesh::new(
        g,
        Material::Standard(StandardMaterial::new(Color::WHITE)),
        skeleton,
    ));
    obj.cast_shadow = true;
    let id = scene.add(obj);
    let updates = |hr: &mut HeadlessRenderer| hr.renderer().shadow_map_updates();

    hr.render(&mut scene, &camera(0.0));
    let first = updates(&mut hr);
    for x in [0.5, 1.0] {
        hr.render(&mut scene, &camera(x));
    }
    assert_eq!(updates(&mut hr), first, "a still pose reuses the maps");

    if let ObjectKind::SkinnedMesh(sm) = &mut scene.get_mut(id).unwrap().kind {
        Arc::make_mut(&mut sm.skeleton).bone_matrices =
            vec![Matrix4::translation(Vector3::new(1.0, 0.5, 0.0))];
    }
    hr.render(&mut scene, &camera(1.0));
    assert_eq!(updates(&mut hr), first + 1, "a new pose redraws them");
}

#[test]
fn custom_shader_casters_reuse_until_their_inputs_change() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let (mut scene, cube, _) = scene();
    let material = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(u_data.data[0].xyz, 1.0);\n\
         }\n",
    )
    .with_data(vec![[0.25, 0.5, 0.75, 1.0]]);
    if let ObjectKind::Mesh(m) = &mut scene.get_mut(cube).unwrap().kind {
        m.material = Arc::new(Material::Shader(material));
    }
    let updates = |hr: &mut HeadlessRenderer| hr.renderer().shadow_map_updates();

    hr.render(&mut scene, &camera(0.0));
    let first = updates(&mut hr);
    for x in [0.5, 1.0] {
        hr.render(&mut scene, &camera(x));
    }
    assert_eq!(updates(&mut hr), first, "unchanged inputs reuse the maps");

    if let ObjectKind::Mesh(m) = &mut scene.get_mut(cube).unwrap().kind {
        if let Material::Shader(s) = Arc::make_mut(&mut m.material) {
            s.data[0][0] = 1.0;
        }
    }
    hr.render(&mut scene, &camera(1.0));
    assert_eq!(updates(&mut hr), first + 1, "new uniform data redraws them");
}

/// The Metal renderer reuses its maps by the same rule.
#[cfg(feature = "metal")]
#[test]
fn metal_reuses_shadow_maps_until_they_would_change() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(W, W).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let (mut scene, cube, light) = scene();
    let render = |r: &mut MetalHeadlessRenderer, scene: &mut Scene, x: f32| {
        r.render_to_rgba(scene, &camera(x)).expect("metal");
        r.renderer().shadow_map_updates()
    };
    let first = render(&mut r, &mut scene, 0.0);
    assert_eq!(first, 1, "the first render draws them");
    for x in [0.5, 1.0] {
        assert_eq!(
            render(&mut r, &mut scene, x),
            first,
            "a moving camera reuses them"
        );
    }
    scene.get_mut(cube).unwrap().position.x = 1.0;
    assert_eq!(
        render(&mut r, &mut scene, 1.0),
        first + 1,
        "a moved caster redraws them"
    );
    assert_eq!(
        render(&mut r, &mut scene, 1.5),
        first + 1,
        "and then they are reused"
    );
    if let ObjectKind::Light(Light::Directional(l)) = &mut scene.get_mut(light).unwrap().kind {
        l.direction = Vector3::new(0.5, -1.0, -0.3).normalize();
    }
    assert_eq!(
        render(&mut r, &mut scene, 1.5),
        first + 2,
        "a turned light redraws them"
    );
}

/// Per pass, not all or nothing: a camera-following cascaded sun redraws its
/// cascades as the camera moves, and a still point light's six faces are
/// left alone. And a caster is drawn only into the passes whose light
/// frustum it is in.
#[test]
fn passes_are_reused_one_by_one_and_casters_culled_per_pass() {
    use threers::lights::PointLight;
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, W).build() else {
        return;
    };
    let (mut scene, _, light) = scene();
    if let ObjectKind::Light(Light::Directional(l)) = &mut scene.get_mut(light).unwrap().kind {
        l.shadow.cascades = 4;
        l.shadow.cascade_distance = 30.0;
    }
    let mut lamp = PointLight::new(Color::WHITE, 20.0);
    lamp.cast_shadow = true;
    lamp.distance = 30.0;
    let mut lamp = Object3D::light(Light::Point(lamp));
    lamp.position = Vector3::new(3.0, 2.0, 0.0);
    scene.add(lamp);
    let stats = |hr: &mut HeadlessRenderer| hr.renderer().shadow_stats();

    hr.render(&mut scene, &camera(0.0));
    let (passes, draws) = stats(&mut hr);
    assert_eq!(passes, 4 + 6, "the first render draws every cascade and face");
    // One cube, beside the lamp: it falls in some of the cube's faces only.
    assert!(draws < 4 + 6, "{draws} caster draws: the cube was drawn into every face");

    hr.render(&mut scene, &camera(1.0));
    let (passes2, _) = stats(&mut hr);
    assert_eq!(passes2 - passes, 4, "a moving camera redraws the cascades and not the lamp's faces");

    hr.render(&mut scene, &camera(1.0));
    assert_eq!(stats(&mut hr).0, passes2, "a still frame redraws nothing");
}
