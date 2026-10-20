//! `threers::volume`: slices through float and label volumes, read back and
//! checked texel by texel. Skips (passes) without a GPU.

use threers::gpu::testing;
use threers::math::{Quaternion, Vector3};
use threers::renderer::{fullscreen, readback, PixelRect};
use threers::volume::{
    plane_to_voxel, Data3DTexture, GpuVolume, SliceParams, SliceRenderer, SliceTarget, VolumeSample,
};

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume slice test target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Clears to transparent, draws one slice over the whole target, reads back.
fn render(
    gpu: &threers::gpu::GpuContext,
    slices: &SliceRenderer,
    volume: &GpuVolume,
    params: &SliceParams,
) -> Vec<u8> {
    let (texture, view) = target(&gpu.device);
    let binding = slices.create_binding(&gpu.device, volume);
    slices.update(&gpu.queue, &binding, params);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    drop(fullscreen::begin(&mut encoder, &view, fullscreen::Load::Clear));
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: SIZE,
        height: SIZE,
    };
    slices.draw(&mut encoder, &view, rect, &binding);
    gpu.queue.submit([encoder.finish()]);
    readback::texture_bytes(&gpu.device, &gpu.queue, &texture)
}

/// The slice through z = 1.5 of a 4×4×4 volume, filling the target: each
/// voxel is a 16 px square, voxel (0, 0) at the top left.
fn whole_slice() -> threers::math::Matrix4 {
    plane_to_voxel(
        Vector3::new(2.0, 2.0, 1.5),
        Quaternion::identity(),
        4.0 / SIZE as f32,
        (SIZE as f32, SIZE as f32),
        true,
    )
}

/// The pixel at the middle of voxel `(x, y)`'s square.
fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let (px, py) = (x * 16 + 8, y * 16 + 8);
    let i = ((py * SIZE + px) * 4) as usize;
    rgba[i..i + 4].try_into().unwrap()
}

#[test]
fn window_level_maps_values_to_grey() {
    let Some(gpu) = testing::context("volume window/level") else {
        return;
    };
    // value = x / 3, so the four columns are 0, 1/3, 2/3, 1.
    let values: Vec<f32> = (0..64).map(|i| (i % 4) as f32 / 3.0).collect();
    let volume = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::from_f32(4, 4, 4, &values));
    let slices = SliceRenderer::window_level(&gpu.device, FORMAT);

    let rgba = render(&gpu, &slices, &volume, &SliceParams::window(whole_slice(), 0.0, 1.0));
    for x in 0..4 {
        let expected = (x as f32 / 3.0 * 255.0).round() as u8;
        let p = pixel(&rgba, x, 2);
        assert!(p[0].abs_diff(expected) <= 1 && p[0] == p[1] && p[1] == p[2], "x={x}: {p:?}");
        assert_eq!(p[3], 255);
    }

    // A narrower window saturates: 1/3 and below are black, 2/3 and up white.
    let rgba = render(&gpu, &slices, &volume, &SliceParams::window(whole_slice(), 0.4, 0.6));
    assert_eq!(pixel(&rgba, 1, 0)[0], 0);
    assert_eq!(pixel(&rgba, 2, 0)[0], 255);
}

#[test]
fn write_region_replaces_only_its_box() {
    let Some(gpu) = testing::context("volume write_region") else {
        return;
    };
    let volume = GpuVolume::from_texture(
        &gpu.device,
        &gpu.queue,
        &Data3DTexture::from_f32(4, 4, 4, &[0.0; 64]),
    );
    // A 2×1×1 strip at (1, 3, 1): on the drawn slice (z = 1).
    volume.write_region(&gpu.queue, [1, 3, 1], [2, 1, 1], bytemuck::cast_slice(&[1.0f32, 0.5]));
    let slices = SliceRenderer::window_level(&gpu.device, FORMAT);
    let rgba = render(&gpu, &slices, &volume, &SliceParams::window(whole_slice(), 0.0, 1.0));

    assert_eq!(pixel(&rgba, 1, 3)[0], 255);
    assert!(pixel(&rgba, 2, 3)[0].abs_diff(128) <= 1);
    for (x, y) in [(0, 3), (3, 3), (1, 2), (0, 0)] {
        assert_eq!(pixel(&rgba, x, y)[0], 0, "({x}, {y}) should be untouched");
    }
}

#[test]
fn labels_color_each_value_and_leave_zero_clear() {
    let Some(gpu) = testing::context("volume labels") else {
        return;
    };
    // Columns: background, label 7, label 7 again, and 7 + 2^32 (same low word).
    let column = [0u64, 7, 7, (1 << 32) | 7];
    let labels: Vec<u64> = (0..64).map(|i| column[i % 4]).collect();
    let volume = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::from_u64(4, 4, 4, &labels));
    let slices = SliceRenderer::labels(&gpu.device, FORMAT);
    let rgba = render(&gpu, &slices, &volume, &SliceParams::labels(whole_slice(), 0, 1.0));

    assert_eq!(pixel(&rgba, 0, 1), [0, 0, 0, 0], "label 0 is transparent");
    assert_eq!(pixel(&rgba, 1, 1)[3], 255);
    assert_eq!(pixel(&rgba, 1, 1), pixel(&rgba, 2, 1), "one label, one colour");
    assert_ne!(pixel(&rgba, 1, 1), pixel(&rgba, 3, 1), "the high word counts");

    // A different seed reshuffles the colours.
    let reseeded = render(&gpu, &slices, &volume, &SliceParams::labels(whole_slice(), 12345, 1.0));
    assert_ne!(pixel(&rgba, 1, 1), pixel(&reseeded, 1, 1));
}

#[test]
fn a_custom_color_function_sees_both_words_of_a_u64_exactly() {
    let Some(gpu) = testing::context("volume custom colour") else {
        return;
    };
    let labels = vec![(5u64 << 32) | 200; 64];
    let volume = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::from_u64(4, 4, 4, &labels));
    let slices = SliceRenderer::with_color(
        &gpu.device,
        FORMAT,
        VolumeSample::Uint,
        r#"
        fn slice_color(value : vec4<u32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
            return vec4<f32>(f32(value.r) / 255.0, f32(value.g) / 255.0, params[1].z, 1.0);
        }
        "#,
    );
    let mut params = SliceParams::labels(whole_slice(), 0, 1.0);
    params.params[1][2] = 1.0;
    let rgba = render(&gpu, &slices, &volume, &params);
    assert_eq!(pixel(&rgba, 2, 2), [200, 5, 255, 255]);
}

#[test]
#[should_panic(expected = "cannot be drawn")]
fn a_float_renderer_refuses_a_label_volume() {
    let Some(gpu) = testing::context("volume sample mismatch") else {
        panic!("cannot be drawn (no GPU; nothing to check)");
    };
    let volume = GpuVolume::new(&gpu.device, [2, 2, 2], threers::volume::VolumeFormat::R32Uint);
    SliceRenderer::window_level(&gpu.device, FORMAT).create_binding(&gpu.device, &volume);
}

/// Both built-in colour functions, translated to GLSL ES 3.0 the way wgpu's
/// WebGL2 backend translates them. The float path is also exercised in a
/// browser by neuroglancer-rs; this covers the integer one, whose
/// `usampler3D` only WebGL2 runs through GLSL.
#[test]
fn built_in_slices_translate_for_webgl2() {
    use threers::volume::{LABELS, WINDOW_LEVEL};
    use wgpu::naga;
    for (name, sample, color) in [
        ("window/level", VolumeSample::Float, WINDOW_LEVEL),
        ("labels", VolumeSample::Uint, LABELS),
    ] {
        let source = SliceRenderer::shader_source(sample, color);
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for (stage, entry) in [(naga::ShaderStage::Vertex, "vs_main"), (naga::ShaderStage::Fragment, "fs")] {
            let options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::Embedded { version: 300, is_webgl: true },
                ..Default::default()
            };
            let pipeline = naga::back::glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.to_string(),
                multiview: None,
            };
            let mut out = String::new();
            naga::back::glsl::Writer::new(&mut out, &module, &info, &options, &pipeline, Default::default())
                .and_then(|mut w| w.write().map(|_| ()))
                .unwrap_or_else(|e| panic!("{name} {entry}: {e}"));
            if sample == VolumeSample::Uint && stage == naga::ShaderStage::Fragment {
                assert!(out.contains("usampler3D"), "{name}: expected an integer 3D sampler:\n{out}");
            }
        }
    }
}

/// With a depth attachment the first draw to reach a pixel keeps it, and with
/// no blending a draw replaces colour and alpha alike.
#[test]
fn depth_target_keeps_the_first_draw_and_unblended_keeps_alpha() {
    let Some(gpu) = testing::context("volume slice depth") else {
        return;
    };
    const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    // Grey `value`, alpha one half.
    const HALF_ALPHA: &str = r#"
fn slice_color(value : vec4<f32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    return vec4<f32>(value.r, value.r, value.r, 0.5);
}
"#;
    let slices = SliceRenderer::with_target(
        &gpu.device,
        SliceTarget {
            format: FORMAT,
            blend: None,
            depth: Some((DEPTH, wgpu::CompareFunction::Less)),
        },
        VolumeSample::Float,
        HALF_ALPHA,
    );
    let fine = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::from_f32(4, 4, 4, &[1.0; 64]));
    let coarse = GpuVolume::from_texture(&gpu.device, &gpu.queue, &Data3DTexture::from_f32(4, 4, 4, &[0.0; 64]));
    let (texture, view) = target(&gpu.device);
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume slice test depth"),
        size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    let (b_fine, b_coarse) = (slices.create_binding(&gpu.device, &fine), slices.create_binding(&gpu.device, &coarse));
    slices.update(&gpu.queue, &b_fine, &SliceParams::window(whole_slice(), 0.0, 1.0));
    slices.update(&gpu.queue, &b_coarse, &SliceParams::window(whole_slice(), 0.0, 1.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // The fine data covers the left half; the coarse data everything.
        let half = PixelRect { x: 0, y: 0, width: SIZE / 2, height: SIZE };
        let whole = PixelRect { x: 0, y: 0, width: SIZE, height: SIZE };
        slices.draw_in_pass(&mut pass, half, &b_fine);
        slices.draw_in_pass(&mut pass, whole, &b_coarse);
    }
    gpu.queue.submit([encoder.finish()]);
    let rgba = readback::texture_bytes(&gpu.device, &gpu.queue, &texture);
    let at = |x: u32, y: u32| -> [u8; 4] { rgba[((y * SIZE + x) * 4) as usize..][..4].try_into().unwrap() };
    // Alpha 0.5 is 127.5 in eight bits, a tie each GPU breaks its own way:
    // 128 on Apple's, 127 on NVIDIA's.
    let half = |p: [u8; 4]| [p[0], p[1], p[2], (p[3] == 127 || p[3] == 128) as u8];
    assert_eq!(half(at(8, 30)), [255, 255, 255, 1], "the fine draw stays: {:?}", at(8, 30));
    assert_eq!(half(at(50, 30)), [0, 0, 0, 1], "the coarse draw fills the rest: {:?}", at(50, 30));
}

/// A colour function can read a lookup table: here a palette indexed by label.
#[test]
fn a_lookup_table_reaches_the_color_function() {
    let Some(gpu) = testing::context("volume slice lookup") else {
        return;
    };
    const PALETTE: &str = r#"
fn slice_color(value : vec4<u32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    let c = textureLoad(slice_lookup, vec2<i32>(i32(value.r), 0), 0).r;
    return vec4<f32>(f32(c & 0xffu) / 255.0, f32((c >> 8u) & 0xffu) / 255.0, f32((c >> 16u) & 0xffu) / 255.0, 1.0);
}
"#;
    let slices = SliceRenderer::with_lookup(
        &gpu.device,
        SliceTarget { format: FORMAT, blend: None, depth: None },
        VolumeSample::Uint,
        PALETTE,
    );
    // Labels = x, so the four columns read palette entries 0..4.
    let labels: Vec<u32> = (0..64).map(|i| i % 4).collect();
    let data = Data3DTexture::new(4, 4, 4, threers::volume::VolumeFormat::R32Uint, bytemuck::cast_slice(&labels).to_vec());
    let volume = GpuVolume::from_texture(&gpu.device, &gpu.queue, &data);
    let palette: [u32; 4] = [0x0000ff, 0x00ff00, 0xff0000, 0x808080];
    let table = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("palette"),
        size: wgpu::Extent3d { width: 4, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        table.as_image_copy(),
        bytemuck::cast_slice(&palette),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16), rows_per_image: None },
        wgpu::Extent3d { width: 4, height: 1, depth_or_array_layers: 1 },
    );
    let (texture, view) = target(&gpu.device);
    let binding = slices.create_binding_with_lookup(&gpu.device, &volume, &table.create_view(&Default::default()));
    slices.update(&gpu.queue, &binding, &SliceParams::window(whole_slice(), 0.0, 1.0));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    drop(fullscreen::begin(&mut encoder, &view, fullscreen::Load::Clear));
    slices.draw(&mut encoder, &view, PixelRect { x: 0, y: 0, width: SIZE, height: SIZE }, &binding);
    gpu.queue.submit([encoder.finish()]);
    let rgba = readback::texture_bytes(&gpu.device, &gpu.queue, &texture);
    assert_eq!(pixel(&rgba, 0, 1), [255, 0, 0, 255]);
    assert_eq!(pixel(&rgba, 1, 1), [0, 255, 0, 255]);
    assert_eq!(pixel(&rgba, 2, 1), [0, 0, 255, 255]);
    assert_eq!(pixel(&rgba, 3, 1), [128, 128, 128, 255]);
}
