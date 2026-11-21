//! A 3D view drawn on a render thread, shown in a UI box by a [`SurfaceView`]
//! and composited by the [`UiPass`]: the view's pixels land inside the box,
//! the UI's around it, and a new frame from the thread shows in the next UI
//! frame.
//!
//! Skipped when no GPU adapter is available.

use std::time::{Duration, Instant};

use threers::gpu::GpuContext;
use threers::renderer::{RendererOptions, UiPass, ViewProducer};
use threers::style;
use threers::ui::surface_view::SurfaceView;
use threers::ui::Ui;
use threers::{Color, PerspectiveCamera, Scene};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const W: u32 = 256;
const H: u32 = 100;

fn read(gpu: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let row = W * 4;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    buffer.slice(..).get_mapped_range().expect("mapped").to_vec()
}

fn px(img: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2]]
}

fn near(a: [u8; 3], b: [u8; 3]) -> bool {
    (0..3).all(|i| a[i].abs_diff(b[i]) <= 2)
}

#[test]
fn a_threaded_view_shows_inside_its_box() {
    let gpu = match GpuContext::headless_blocking() {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let (mut producer, frames) = ViewProducer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        RendererOptions::default(),
        None,
    );
    // Wake the "UI" when a frame lands.
    let (tx, rx) = std::sync::mpsc::channel();
    frames.on_frame(move || {
        let _ = tx.send(());
    });
    let (colour_tx, colour_rx) = std::sync::mpsc::channel::<u32>();
    let render = std::thread::spawn(move || {
        let camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        let mut scene = Scene::new();
        while producer.wait_for_work() {
            if let Ok(hex) = colour_rx.try_recv() {
                scene.background = Color::from_hex(hex);
            }
            producer.render(&mut scene, &camera);
        }
    });

    // The view fills the right half; the left is the UI's background.
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: row; background: #0000ff; });
    ui.add_box(root, style! { width: 128px; });
    let mut view = SurfaceView::new(&mut ui, root, style! { flex-grow: 1; }, frames);
    let mut pass = UiPass::new(&gpu.device);

    let mut ui_frame = |colour: u32| -> Vec<u8> {
        colour_tx.send(colour).expect("render thread");
        // A UI frame lays out and asks; a frame lands; the next UI frame shows it.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
            let shown = view.update(&ui, &mut pass, 1.0);
            pass.draw(&gpu.device, &gpu.queue, &ui.scene(), 1.0, &target_view, (W, H), FORMAT);
            let img = read(&gpu, &target);
            let want = [(colour >> 16) as u8, (colour >> 8) as u8, colour as u8];
            if shown.is_some() && near(px(&img, 200, 50), want) {
                return img;
            }
            assert!(Instant::now() < deadline, "the view never showed {colour:06x}");
            view.frames().request_redraw();
            let _ = rx.recv_timeout(Duration::from_millis(500));
        }
    };

    let img = ui_frame(0xff0000);
    assert_eq!(px(&img, 200, 50), [255, 0, 0]);
    assert_eq!(px(&img, 20, 50), [0, 0, 255], "the UI around the box");
    let img = ui_frame(0x00ff00);
    assert!(near(px(&img, 129, 1), [0, 255, 0]), "the box's corner is the view's");
    assert_eq!(px(&img, 20, 50), [0, 0, 255]);

    drop(view);
    render.join().expect("render thread");
}

#[test]
fn pointer_points_map_into_the_view_unless_covered() {
    let gpu = match GpuContext::headless_blocking() {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let (_producer, frames) = ViewProducer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        RendererOptions::default(),
        None,
    );
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { position: relative; });
    let view = SurfaceView::new(
        &mut ui,
        root,
        style! { position: absolute; left: 100px; top: 50px; width: 200px; height: 100px; },
        frames,
    );
    // A badge over the view's top-left corner.
    ui.add_box(
        root,
        style! { position: absolute; left: 100px; top: 50px; width: 20px; height: 20px; background: #fff; },
    );
    let mut pass = UiPass::new(&gpu.device);
    ui.layout(400.0, 300.0, &mut pass.text_measure(1.0));

    let p = view.point(&ui, 200.0, 75.0).expect("inside");
    assert_eq!(p.uv, [0.5, 0.25]);
    assert_eq!(p.ndc, [0.0, 0.5]);
    let corner = view.point(&ui, 299.0, 149.0).expect("inside");
    assert!(corner.ndc[0] > 0.98 && corner.ndc[1] < -0.97);
    assert_eq!(view.point(&ui, 50.0, 75.0), None, "outside");
    assert_eq!(view.point(&ui, 105.0, 55.0), None, "under the badge");
    assert_eq!(view.pixel_size(&ui, 2.0), (400, 200));
}

#[test]
fn a_box_that_cannot_be_seen_is_not_drawn() {
    let gpu = match GpuContext::headless_blocking() {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let (_producer, frames) = ViewProducer::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        FORMAT,
        RendererOptions::default(),
        None,
    );
    let mut ui = Ui::new();
    let root = ui.root();
    // A clipping list 100 px tall; the view is its second 80 px row.
    let list = ui.add_box(root, style! { height: 100px; overflow: hidden; });
    ui.add_box(list, style! { height: 80px; });
    let mut view = SurfaceView::new(&mut ui, list, style! { height: 80px; }, frames);
    let mut pass = UiPass::new(&gpu.device);
    let mut seen = |ui: &mut Ui, view: &mut SurfaceView| {
        ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
        view.update(ui, &mut pass, 1.0);
        view.frames().is_visible()
    };
    // 20 px of it inside the list: seen, and only that part.
    assert!(seen(&mut ui, &mut view));
    let r = view.visible_rect(&ui).expect("partly seen");
    assert_eq!((r.y, r.height), (80.0, 20.0));
    // Pushed below the list's bottom: clipped away entirely.
    ui.set_style(ui.children(list)[0], style! { height: 120px; });
    assert!(!seen(&mut ui, &mut view));
    // Back in, then hidden by an ancestor's display.
    ui.set_style(ui.children(list)[0], style! { height: 0px; });
    assert!(seen(&mut ui, &mut view));
    ui.set_style(list, style! { height: 100px; overflow: hidden; display: none; });
    assert!(!seen(&mut ui, &mut view));
}
