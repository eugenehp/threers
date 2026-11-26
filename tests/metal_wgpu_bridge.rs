//! wgpu on the native Metal backend's own device and queue
//! (`metal::wgpu_on`): a Metal scene, then the GPU UI drawn over it through
//! wgpu in the same queue, read back as one picture.
#![cfg(metal_backend)]

use std::sync::Arc;

use threers::cameras::PerspectiveCamera;
use threers::math::Color;
use threers::metal::{wgpu_on, wgpu_texture, MetalDevice, MetalError, MetalHeadlessRenderer};
use threers::renderer::UiPass;
use threers::scene::Scene;
use threers::ui::{style, Ui};

const N: u32 = 64;

#[test]
fn the_ui_draws_over_a_metal_scene_in_one_queue() {
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(MetalError::NoDevice) => return eprintln!("skipping: no Metal device"),
        Err(e) => panic!("{e}"),
    };
    let mut metal = MetalHeadlessRenderer::builder().size(N, N).msaa(1).build_on(&device).unwrap();
    // the scene: red all over
    let mut scene = Scene::new();
    scene.background = Color::new(1.0, 0.0, 0.0);
    let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    metal.render(&mut scene, &camera).unwrap();
    // wgpu on the same device and queue, once
    let (gpu, queue) = wgpu_on(&device).expect("wgpu on Metal");
    let (again, _) = wgpu_on(&device).unwrap();
    assert!(Arc::ptr_eq(&gpu, &again), "one wgpu device for one Metal device");
    // the UI: a white box over the left half, drawn into the Metal target
    let mut ui = Ui::new();
    let root = ui.root();
    ui.add_box(root, style! { width: 32px; height: 64px; background: #ffffff; });
    let mut pass = UiPass::new(&gpu);
    ui.layout(N as f32, N as f32, &mut pass.text_measure(1.0));
    // SAFETY: the target's colour texture, on this device, alive meanwhile.
    let texture = unsafe { wgpu_texture(&gpu, metal.target().color_texture(), wgpu::TextureUsages::RENDER_ATTACHMENT) }.expect("a format wgpu names");
    let view = texture.create_view(&Default::default());
    pass.draw(&gpu, &queue, &ui.scene(), 1.0, &view, (N, N), texture.format());
    gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    // read on the Metal side: what both drew, in the order they were sent
    let px = metal.target().read_rgba(&device).unwrap();
    let at = |x: u32, y: u32| &px[((y * N + x) * 4) as usize..][..4];
    assert_eq!(at(8, 32), [255, 255, 255, 255], "the UI's box, over the scene");
    let right = at(56, 32);
    assert!(right[0] > 240 && right[1] < 15 && right[2] < 15, "the scene beside it: {right:?}");
}

/// The wgpu scene renderer, on the Metal device's wgpu: what a window on
/// the native backend shares with views drawn on other threads
/// (`WindowRenderer::shared_gpu`).
#[test]
fn a_view_renders_with_wgpu_on_the_metal_device() {
    use threers::renderer::{SharedGpu, ViewOptions, ViewProducer};
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(MetalError::NoDevice) => return eprintln!("skipping: no Metal device"),
        Err(e) => panic!("{e}"),
    };
    let (gpu, queue) = wgpu_on(&device).unwrap();
    let (mut producer, mut frames) = ViewProducer::on(&SharedGpu::new(gpu.clone(), queue), ViewOptions::default());
    frames.set_size(N, N);
    let mut scene = Scene::new();
    scene.background = Color::new(0.0, 0.0, 1.0);
    let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    assert!(producer.render(&mut scene, &camera), "drawn");
    gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let frame = frames.latest().expect("a frame for the window to show");
    assert_eq!(frame.size, (N, N));
}
