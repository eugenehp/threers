//! `WindowRenderer` on the native Metal backend without winit: a
//! `CAMetalLayer` of the app's own — the way in on iOS, tvOS and visionOS
//! apps — drawing a scene and marks over it, on every Apple platform.
//!
//! The layer is attached to nothing, so the drawables never reach a screen;
//! what is tested is that every frame is drawn, marks included, and
//! presented without a Metal error.

#![cfg(metal_backend)]

use threers::marks::{Mark, MarkRenderer, MarkView, Origin, PointMarks};
use threers::metal::objc::{alloc_init, get_class, Owned};
use threers::renderer::WindowRenderer;
use threers::{BoxGeometry, Color, Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3};

#[test]
fn a_metal_layer_window_draws_a_scene_and_marks() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    // SAFETY: a fresh CAMetalLayer, kept alive for the whole test.
    let layer = unsafe { Owned::from_retained(alloc_init(get_class("CAMetalLayer\0"))) }.expect("a CAMetalLayer");
    let mut wr = unsafe { WindowRenderer::from_layer(layer.id().cast(), 96, 64) }.expect("a Metal window renderer");
    assert!(wr.is_native_metal());
    assert_eq!(wr.size(), (96, 64));

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x204060);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xff8844)).into(),
    )));
    let mut camera = PerspectiveCamera::new(50.0, wr.aspect(), 0.1, 100.0);
    camera.position = Vector3::new(2.0, 1.5, 3.0);
    camera.look_at(Vector3::ZERO);

    let mut marks = MarkRenderer::new_on(&wr.marks_gpu());
    let mut points = PointMarks::new();
    points.set_positions(&[[0.0, 0.0, 0.0], [0.6, 0.6, 0.0]]);
    points.style_mut().size_scale = 6.0;

    let mut drawn = 0;
    for msaa in [4, 1] {
        wr.set_msaa(msaa);
        for _ in 0..3 {
            let mut over = false;
            let ok = wr.render_with(&mut scene, &camera, |f| {
                assert_eq!(f.surface.size, (96, 64));
                assert_eq!(f.target.samples, msaa);
                // Single-sampled frames can be picked against.
                assert_eq!(f.occluder.is_some(), msaa == 1);
                let view = MarkView::for_surface(&camera, Origin::ZERO, &f.surface);
                marks.draw_on(&f.gpu, &f.target, &view, &mut [&mut points as &mut dyn Mark]);
                over = true;
            });
            if ok {
                assert!(over, "the marks callback runs on every frame drawn");
                drawn += 1;
            }
            wr.poll();
        }
        // Plain frames still draw on the same renderer.
        wr.render(&mut scene, &camera);
    }
    assert!(drawn >= 4, "only {drawn} of 6 frames were drawn");
}
