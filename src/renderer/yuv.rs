//! Video frames converted on the GPU: a decoder's YUV planes in, the RGBA
//! texture a material samples out.
//!
//! A [`TextureStream`](crate::textures::TextureStream) written with
//! [`YuvFrame`]s hands the renderer the planes as the decoder made them, one
//! `u8` or `u16` a sample. They go up as three `R8Uint` or `R16Uint` textures,
//! and one fullscreen
//! pass converts, turns the picture over (`flip_y`) and, when the frame is
//! larger than the device can hold, shrinks it by a whole factor, averaging
//! in linear light. The arithmetic is [`YuvFrame::to_rgba_into`]'s, so the
//! GPU and the CPU agree to a level.
//!
//! At 8-bit 4:2:0 the planes are 3 bytes a pixel against RGBA's 4, and no CPU
//! pass touches every pixel: at 4K and 8K that pass was most of the cost of
//! showing a frame.

use crate::renderer::fullscreen::{FormatPipelines, VS, VS_ENTRY};
#[cfg(test)]
use crate::textures::Samples;
use crate::textures::YuvFrame;
use std::collections::HashMap;

/// `YuvParams` in the shader, field for field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct YuvParams {
    /// Luma width and height, chroma width and height.
    pub dims: [u32; 4],
    /// Chroma halved across, halved down, turned over, luma only.
    pub mode: [u32; 4],
    /// Source pixels a target pixel each way, sRGB target, target height,
    /// chroma interleaved (Cb and Cr in one two-channel plane).
    pub out: [u32; 4],
    /// Luma offset and scale, chroma midpoint.
    pub k0: [f32; 4],
    /// Cr into red, Cb and Cr out of green, Cb into blue.
    pub k1: [f32; 4],
}

impl YuvParams {
    /// For `frame` drawn into a `target_height` texture `scale` times smaller.
    pub(crate) fn new(
        frame: &YuvFrame,
        flip: bool,
        scale: u32,
        srgb_target: bool,
        target_height: u32,
    ) -> Self {
        let k = frame.coefficients();
        let (cw, ch) = frame.chroma_size();
        Self {
            dims: [frame.width, frame.height, cw.max(1), ch.max(1)],
            mode: [
                (frame.subsampling.0 == 2) as u32,
                (frame.subsampling.1 == 2) as u32,
                flip as u32,
                frame.is_monochrome() as u32,
            ],
            out: [
                scale.max(1),
                srgb_target as u32,
                target_height.max(1),
                frame.interleaved as u32,
            ],
            k0: [k.y_off, k.y_scale, k.c_mid, 0.0],
            k1: [k.rv, k.gu, k.gv, k.bu],
        }
    }
}

/// The smallest whole factor that brings `width × height` within `max` each
/// way, and the size it gives.
pub(crate) fn fit(width: u32, height: u32, max: u32) -> (u32, u32, u32) {
    let max = max.max(1);
    let scale = width.div_ceil(max).max(height.div_ceil(max)).max(1);
    (
        scale,
        width.div_ceil(scale).max(1),
        height.div_ceil(scale).max(1),
    )
}

const FS: &str = crate::wgsl!(
    r#"
struct YuvParams {
    dims : vec4<u32>,
    mode : vec4<u32>,
    out : vec4<u32>,
    k0 : vec4<f32>,
    k1 : vec4<f32>,
};

@group(0) @binding(0) var y_plane : texture_2d<u32>;
@group(0) @binding(1) var u_plane : texture_2d<u32>;
@group(0) @binding(2) var v_plane : texture_2d<u32>;
@group(0) @binding(3) var<uniform> p : YuvParams;

fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn plane_sample(which : u32, x : u32, y : u32) -> f32 {
    if (p.out.w == 1u) {
        // NV12/P010: Cb and Cr side by side in one plane.
        let pair = textureLoad(u_plane, vec2<u32>(x, y), 0);
        return f32(select(pair.r, pair.g, which == 1u));
    }
    if (which == 0u) {
        return f32(textureLoad(u_plane, vec2<u32>(x, y), 0).r);
    }
    return f32(textureLoad(v_plane, vec2<u32>(x, y), 0).r);
}

// Chroma at luma (x, y): bilinear from where H.264, HEVC and AV1 site it by
// default, the weights summing to 8, less the midpoint.
fn chroma(which : u32, x : u32, y : u32) -> f32 {
    let cw = p.dims.z;
    let ch = p.dims.w;
    var r0 = min(y, ch - 1u);
    var r1 = r0;
    var wr0 = 4.0;
    var wr1 = 0.0;
    if (p.mode.y == 1u) {
        r0 = min(y / 2u, ch - 1u);
        if (y % 2u == 0u) {
            r1 = select(r0 - 1u, 0u, r0 == 0u);
        } else {
            r1 = min(r0 + 1u, ch - 1u);
        }
        wr0 = 3.0;
        wr1 = 1.0;
    }
    var c0 = min(x, cw - 1u);
    var c1 = c0;
    var wc0 = 2.0;
    var wc1 = 0.0;
    if (p.mode.x == 1u) {
        c0 = x / 2u;
        c1 = c0;
        if (x % 2u == 1u) {
            c1 = min(c0 + 1u, cw - 1u);
            wc0 = 1.0;
            wc1 = 1.0;
        }
    }
    let sum = wr0 * (wc0 * plane_sample(which, c0, r0) + wc1 * plane_sample(which, c1, r0))
        + wr1 * (wc0 * plane_sample(which, c0, r1) + wc1 * plane_sample(which, c1, r1));
    return sum / 8.0 - p.k0.z;
}

// The picture's colour at (x, y), sRGB-encoded, 0 to 1.
fn rgb_at(x : u32, y : u32) -> vec3<f32> {
    let l = p.k0.y * (f32(textureLoad(y_plane, vec2<u32>(x, y), 0).r) - p.k0.x);
    var cb = 0.0;
    var cr = 0.0;
    if (p.mode.w == 0u) {
        cb = chroma(0u, x, y);
        cr = chroma(1u, x, y);
    }
    let rgb = vec3<f32>(l + p.k1.x * cr, l - p.k1.y * cb - p.k1.z * cr, l + p.k1.w * cb);
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    let px = vec2<u32>(in.clip.xy);
    let k = max(p.out.x, 1u);
    var row = px.y;
    if (p.mode.z == 1u) {
        row = p.out.z - 1u - px.y;
    }
    if (k == 1u) {
        let c = rgb_at(min(px.x, p.dims.x - 1u), min(row, p.dims.y - 1u));
        if (p.out.y == 1u) {
            return vec4<f32>(srgb_to_linear(c), 1.0);
        }
        return vec4<f32>(c, 1.0);
    }
    // Shrunk to fit the device: the mean of the block, in linear light.
    var sum = vec3<f32>(0.0);
    for (var j = 0u; j < k; j = j + 1u) {
        for (var i = 0u; i < k; i = i + 1u) {
            let x = min(px.x * k + i, p.dims.x - 1u);
            let y = min(row * k + j, p.dims.y - 1u);
            sum = sum + srgb_to_linear(rgb_at(x, y));
        }
    }
    let lin = sum / f32(k * k);
    if (p.out.y == 1u) {
        return vec4<f32>(lin, 1.0);
    }
    return vec4<f32>(linear_to_srgb(lin), 1.0);
}
"#
);

/// One stream's planes on the GPU, reused while the frame's shape holds.
struct Planes {
    sizes: [(u32, u32); 3],
    formats: [wgpu::TextureFormat; 3],
    textures: [wgpu::Texture; 3],
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// The conversion pass and each stream's planes. See the [module docs](self).
pub(crate) struct YuvConverter {
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipelines: FormatPipelines<wgpu::RenderPipeline>,
    planes: HashMap<u64, Planes>,
}

impl YuvConverter {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers yuv"),
            source: wgpu::ShaderSource::Wgsl(format!("{VS}\n{FS}").into()),
        });
        let plane = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers yuv layout"),
            entries: &[
                plane(0),
                plane(1),
                plane(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        Self {
            shader,
            layout,
            pipelines: FormatPipelines::new(),
            planes: HashMap::new(),
        }
    }

    /// Upload `frame`'s planes and draw them, converted, into level 0 of
    /// `target`, whose format is `format`. `id` names the stream, whose
    /// planes are kept for the next frame. The target may be smaller than the
    /// frame by a whole factor ([`fit`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn convert(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: u64,
        frame: &YuvFrame,
        target: &wgpu::Texture,
        format: wgpu::TextureFormat,
        flip: bool,
    ) {
        if !frame.is_valid() {
            return;
        }
        let (cw, ch) = frame.chroma_size();
        // A luma-only frame binds 1x1 chroma planes the shader never reads;
        // an interleaved one a two-channel plane and a 1x1 spare.
        let chroma = if frame.is_monochrome() {
            (1, 1)
        } else {
            (cw, ch)
        };
        let sizes = if frame.interleaved {
            [(frame.width.max(1), frame.height.max(1)), chroma, (1, 1)]
        } else {
            [(frame.width.max(1), frame.height.max(1)), chroma, chroma]
        };
        let bytes = frame.y.bytes_per_sample();
        let (one, two) = if bytes == 1 {
            (wgpu::TextureFormat::R8Uint, wgpu::TextureFormat::Rg8Uint)
        } else {
            (wgpu::TextureFormat::R16Uint, wgpu::TextureFormat::Rg16Uint)
        };
        let formats = if frame.interleaved {
            [one, two, one]
        } else {
            [one; 3]
        };
        if self
            .planes
            .get(&id)
            .is_none_or(|p| p.sizes != sizes || p.formats != formats)
        {
            let planes = self.make_planes(device, sizes, formats);
            self.planes.insert(id, planes);
        }
        let planes = &self.planes[&id];
        let blank = [0u8; 4];
        let data: [&[u8]; 3] = if frame.is_monochrome() {
            [
                frame.y.as_bytes(),
                &blank[..bytes as usize],
                &blank[..bytes as usize],
            ]
        } else if frame.interleaved {
            [
                frame.y.as_bytes(),
                frame.u.as_bytes(),
                &blank[..bytes as usize],
            ]
        } else {
            [frame.y.as_bytes(), frame.u.as_bytes(), frame.v.as_bytes()]
        };
        for (((texture, &(w, h)), samples), format) in
            planes.textures.iter().zip(&sizes).zip(data).zip(formats)
        {
            let bytes = format.block_copy_size(None).unwrap_or(bytes);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                samples,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * bytes),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let scale = frame
            .width
            .div_ceil(target.width().max(1))
            .max(frame.height.div_ceil(target.height().max(1)));
        let params = YuvParams::new(frame, flip, scale, format.is_srgb(), target.height());
        queue.write_buffer(&planes.params, 0, bytemuck::bytes_of(&params));

        let layout = &self.layout;
        let shader = &self.shader;
        let pipeline = self.pipelines.get_or_build(format, || {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers yuv pipeline layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers yuv"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(VS_ENTRY),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs"),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers yuv"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers yuv pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &planes.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn make_planes(
        &self,
        device: &wgpu::Device,
        sizes: [(u32, u32); 3],
        formats: [wgpu::TextureFormat; 3],
    ) -> Planes {
        let mut at = 0;
        let textures = sizes.map(|(width, height)| {
            let format = formats[at];
            at += 1;
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers yuv plane"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers yuv params"),
            size: std::mem::size_of::<YuvParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let views = textures
            .each_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
        // Interleaved: the Cb/Cr plane at both chroma bindings.
        let second = if formats[1] != formats[0] {
            &views[1]
        } else {
            &views[2]
        };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers yuv bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(second),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        Planes {
            sizes,
            formats,
            textures,
            params,
            bind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::YuvMatrix;

    fn device() -> Option<crate::gpu::GpuContext> {
        crate::gpu::GpuContext::headless_blocking().ok()
    }

    /// Planes with detail in every one: luma ramps across, chroma varies both
    /// ways, so siting and orientation both show.
    fn frame(
        w: u32,
        h: u32,
        depth: u8,
        sub: (u8, u8),
        mono: bool,
        matrix: YuvMatrix,
        full: bool,
    ) -> YuvFrame {
        let max = (1u32 << depth) - 1;
        let y: Vec<u16> = (0..w * h)
            .map(|i| ((i % w) * max / w.max(1) + (i / w) * 3) as u16 % max as u16)
            .collect();
        let mut f = YuvFrame {
            width: w,
            height: h,
            bit_depth: depth,
            subsampling: sub,
            y: y.into(),
            matrix,
            full_range: full,
            ..Default::default()
        };
        if !mono {
            // From the subsampling: `chroma_size` says (0, 0) until there is
            // chroma.
            let (cw, ch) = (w.div_ceil(sub.0 as u32), h.div_ceil(sub.1 as u32));
            f.u = (0..cw * ch)
                .map(|i| ((i % cw) * max / cw + 7 * (i / cw)) as u16 % max as u16)
                .collect::<Vec<u16>>()
                .into();
            f.v = (0..cw * ch)
                .map(|i| (max - ((i / cw) * max / ch + 5 * (i % cw)) % (max + 1)) as u16)
                .collect::<Vec<u16>>()
                .into();
        }
        assert!(f.is_valid());
        f
    }

    fn convert(
        gpu: &crate::gpu::GpuContext,
        frame: &YuvFrame,
        target: (u32, u32),
        format: wgpu::TextureFormat,
        flip: bool,
    ) -> Vec<u8> {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: target.0,
                height: target.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut yuv = YuvConverter::new(&gpu.device);
        yuv.convert(&gpu.device, &gpu.queue, 1, frame, &texture, format, flip);
        crate::renderer::readback::texture_bytes(&gpu.device, &gpu.queue, &texture)
    }

    fn worst(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn the_gpu_converts_as_the_cpu_does() {
        let Some(gpu) = device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let cases = [
            (8, (2, 2), false, YuvMatrix::Bt709, false),
            (8, (2, 2), false, YuvMatrix::Bt601, true),
            (10, (2, 2), false, YuvMatrix::Bt2020, false),
            (8, (2, 1), false, YuvMatrix::Bt709, false),
            (12, (1, 1), false, YuvMatrix::Bt709, true),
            (8, (1, 2), false, YuvMatrix::Bt601, false),
            (8, (2, 2), true, YuvMatrix::Bt709, false),
        ];
        // Odd sizes, so the last chroma column and row are partly outside.
        for (depth, sub, mono, matrix, full) in cases {
            let f = frame(37, 23, depth, sub, mono, matrix, full);
            if depth == 8 {
                // The same frame a byte a sample converts the same.
                let narrow = |s: &Samples| match s {
                    Samples::Words(v) => Samples::Bytes(v.iter().map(|&x| x as u8).collect()),
                    other => other.clone(),
                };
                let bytes = YuvFrame {
                    y: narrow(&f.y),
                    u: narrow(&f.u),
                    v: narrow(&f.v),
                    ..f.clone()
                };
                assert!(bytes.is_valid());
                let (mut a, mut b) = (Vec::new(), Vec::new());
                f.to_rgba_into(&mut a);
                bytes.to_rgba_into(&mut b);
                assert_eq!(a, b);
                let got = convert(
                    &gpu,
                    &bytes,
                    (37, 23),
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    false,
                );
                assert!(worst(&got, &a) <= 1, "8-bit {sub:?} mono {mono}: bytes");
            }
            let mut want = Vec::new();
            f.to_rgba_into(&mut want);
            for format in [
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm,
            ] {
                let got = convert(&gpu, &f, (37, 23), format, false);
                let d = worst(&got, &want);
                assert!(
                    d <= 1,
                    "{depth}-bit {sub:?} mono {mono} {matrix:?} full {full} {format:?}: off by {d}"
                );
            }
            if sub == (2, 2) && !mono {
                // The same frame with Cb and Cr in pairs (NV12, P010 order).
                let (Samples::Words(u), Samples::Words(v)) = (&f.u, &f.v) else {
                    unreachable!()
                };
                let pairs: Vec<u16> = u.iter().zip(v).flat_map(|(a, b)| [*a, *b]).collect();
                let mut nv = YuvFrame {
                    u: pairs.clone().into(),
                    v: Samples::Words(Vec::new()),
                    interleaved: true,
                    ..f.clone()
                };
                if depth == 8 {
                    let Samples::Words(y) = &f.y else {
                        unreachable!()
                    };
                    nv.y = Samples::Bytes(y.iter().map(|&x| x as u8).collect());
                    nv.u = Samples::Bytes(pairs.iter().map(|&x| x as u8).collect());
                    nv.v = Samples::Bytes(Vec::new());
                }
                assert!(nv.is_valid(), "{depth}-bit interleaved");
                let mut cpu = Vec::new();
                nv.to_rgba_into(&mut cpu);
                assert_eq!(cpu, want, "{depth}-bit interleaved on the CPU");
                let got = convert(
                    &gpu,
                    &nv,
                    (37, 23),
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    false,
                );
                assert!(
                    worst(&got, &want) <= 1,
                    "{depth}-bit interleaved on the GPU"
                );
            }
            // Turned over: the rows reversed, nothing else.
            let got = convert(
                &gpu,
                &f,
                (37, 23),
                wgpu::TextureFormat::Rgba8UnormSrgb,
                true,
            );
            let stride = 37 * 4;
            let turned: Vec<u8> = want.chunks_exact(stride).rev().flatten().copied().collect();
            assert!(
                worst(&got, &turned) <= 1,
                "{depth}-bit {sub:?}: turned over"
            );
        }
    }

    #[test]
    fn a_frame_too_big_for_the_device_is_shrunk_whole() {
        let Some(gpu) = device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        assert_eq!(fit(7680, 4320, 4096), (2, 3840, 2160));
        assert_eq!(fit(1920, 1080, 8192), (1, 1920, 1080));
        assert_eq!(fit(8193, 10, 4096), (3, 2731, 4));
        // Left half one colour, right half another: each output pixel is the
        // mean of a 2x2 block wholly inside one half, so the colours hold.
        let (w, h) = (16u32, 8u32);
        let mut f = frame(w, h, 8, (2, 2), false, YuvMatrix::Bt709, false);
        f.y = (0..w * h)
            .map(|i| if i % w < w / 2 { 60u16 } else { 200 })
            .collect::<Vec<u16>>()
            .into();
        f.u = vec![128u16; f.u.len()].into();
        f.v = vec![128u16; f.v.len()].into();
        let mut full = Vec::new();
        f.to_rgba_into(&mut full);
        let got = convert(
            &gpu,
            &f,
            (w / 2, h / 2),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            false,
        );
        let left = &full[..4];
        let right = &full[((w - 1) * 4) as usize..][..4];
        assert!(worst(&got[..4], left) <= 1, "{:?} vs {left:?}", &got[..4]);
        assert!(worst(&got[((w / 2 - 1) * 4) as usize..][..4], right) <= 1);
    }
}
