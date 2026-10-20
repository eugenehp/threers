//! Render targets as three.js r165 treats them.
//!
//! - **Tone mapping** applies to the final output only (`WebGLRenderer` sets
//!   `toneMapping` to `NoToneMapping` whenever a render target is bound). A
//!   white quad under `Linear` tone mapping at exposure 0.5 is 188 on the
//!   picture (0.5, sRGB-encoded) and 255 in a render target — unless the target
//!   [presents](threers::RenderTarget::presents), standing in for the canvas.
//! - **Read-back** returns what the target's texture colour space says: a
//!   linear 0.5 grey reads 128 from the default (`NoColorSpace`) target and 188
//!   from an `SRGBColorSpace` one.
//!
//! Skipped without a GPU adapter.

use threers::cameras::OrthographicCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::BasicMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::{RenderTarget, ToneMapping};

const N: u32 = 16;

fn scene(grey: f32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.0, 0.0, 0.0);
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        BasicMaterial::new(Color::new(grey, grey, grey)).into(),
    )));
    scene
}

fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

/// The target's raw bytes, tightly packed RGBA.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, rt: &RenderTarget) -> Vec<u8> {
    let padded = (N * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * N) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        rt.color_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(N) },
        },
        wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().expect("mapped").to_vec();
    (0..N).flat_map(|y| data[(y * padded) as usize..(y * padded + N * 4) as usize].to_vec()).collect()
}

fn centre(rgba: &[u8]) -> u8 {
    rgba[((N / 2 * N + N / 2) * 4) as usize]
}

#[test]
fn tone_mapping_and_read_back_follow_three_js() {
    let Ok(mut hr) = threers::HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    hr.renderer().set_tone_mapping(ToneMapping::Linear, 0.5);
    let cam = camera();
    // The picture is tone-mapped.
    assert_eq!(centre(&hr.render_to_rgba(&mut scene(1.0), &cam)), 188, "the picture, tone-mapped");

    let renderer = hr.renderer();
    let (device, queue) = (renderer.device_arc(), renderer.queue_arc());
    let rt = RenderTarget::new(&device, N, N, renderer.color_format());
    renderer.render_to(&mut scene(1.0), &cam, &rt);
    assert_eq!(centre(&read(&device, &queue, &rt)), 255, "a render target stores untone-mapped light");

    rt.set_presents(true);
    renderer.render_to(&mut scene(1.0), &cam, &rt);
    assert_eq!(centre(&read(&device, &queue, &rt)), 188, "a presenting target is the picture, tone-mapped");

    renderer.set_tone_mapping(ToneMapping::None, 1.0);
    rt.set_presents(false);
    renderer.render_to(&mut scene(0.5), &cam, &rt);
    let mut bytes = read(&device, &queue, &rt);
    let raw = bytes.clone();
    rt.bytes_as_color_space(&mut bytes);
    assert_eq!(centre(&bytes), 128, "NoColorSpace reads back linear 0.5");
    rt.set_srgb_color_space(true);
    let mut encoded = raw.clone();
    rt.bytes_as_color_space(&mut encoded);
    assert_eq!(centre(&encoded), 188, "SRGBColorSpace reads back encoded");
    assert_eq!(bytes[3 + ((N / 2 * N + N / 2) * 4) as usize], 255, "alpha untouched");

    let resized = rt.resized(2 * N, N);
    assert_eq!((resized.width, resized.height, resized.format), (2 * N, N, rt.format));
    assert!(resized.srgb_color_space() && !resized.presents(), "setSize keeps the configuration");
}
