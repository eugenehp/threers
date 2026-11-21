//! One multisampled renderer drawing at several sizes in turn, as an adaptive
//! resolution, a resize back and forth or several views on one renderer do:
//! each size draws as it would alone, transparent draws included (they blend
//! through a second view of the multisampled colour, which has to follow the
//! size too).
//!
//! Skipped when no GPU adapter is available.

use threers::gpu::GpuContext;
use threers::renderer::{RenderTarget, Renderer};
use threers::{BasicMaterial, Color, Mesh, Object3D, PerspectiveCamera, PlaneGeometry, Scene, Vector3};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn read(gpu: &GpuContext, tex: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (tex.width(), tex.height());
    let row = (w * 4).next_multiple_of(256);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = buffer.slice(..).get_mapped_range().expect("mapped");
    data.chunks(row as usize).flat_map(|r| r[..(w * 4) as usize].to_vec()).collect()
}

/// An opaque red plane behind a half-transparent white one.
fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x203040);
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.6, 1.6),
        BasicMaterial::new(Color::from_hex(0xff0000)).into(),
    )));
    let mut front = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.8, 0.8),
        BasicMaterial::new(Color::WHITE)
            .with_transparent(true)
            .with_opacity(0.5)
            .into(),
    ));
    front.position = Vector3::new(0.0, 0.0, 0.5);
    scene.add(front);
    scene
}

fn draw(gpu: &GpuContext, r: &mut Renderer, size: (u32, u32)) -> Vec<u8> {
    let mut camera = PerspectiveCamera::new(45.0, size.0 as f32 / size.1 as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));
    // As a view on a shared renderer draws: into a multisampled target of its
    // own, which presents (tone-mapped like a canvas).
    let rt = RenderTarget::new(&gpu.device, size.0, size.1, FORMAT);
    rt.set_presents(true);
    rt.set_samples(4);
    r.render_to(&mut scene(), &camera, &rt);
    read(gpu, &rt.color_texture)
}

#[test]
fn sizes_drawn_in_turn_match_each_drawn_alone() {
    let gpu = match GpuContext::headless_blocking() {
        Ok(g) => g,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let (a, b, c) = ((64, 64), (48, 32), (40, 56));
    let distinct = [a, b, c];
    // Two sizes alternating (a spare taken back) and three in turn (a
    // four-way split's worth of sizes).
    let sizes = [a, b, a, b, c, a, b, c, a];
    let alone: Vec<Vec<u8>> = distinct
        .iter()
        .map(|&s| {
            let mut r = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, s.0, s.1);
            draw(&gpu, &mut r, s)
        })
        .collect();
    let mut r = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, 64, 64);
    for (i, &s) in sizes.iter().enumerate() {
        let img = draw(&gpu, &mut r, s);
        let want = &alone[distinct.iter().position(|&d| d == s).expect("one of them")];
        let worst = img.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        assert!(worst <= 1, "draw {i} at {s:?} differs from the size drawn alone by {worst}");
    }
}
