//! Alpha-blended RGBA overlay pass — how captions reach the screen.
//!
//! A self-contained pipeline that draws one full-screen textured quad over an
//! already-rendered target with `LoadOp::Load`, so the 3D image underneath is
//! preserved and only the caption's own pixels are blended in. It owns nothing
//! from the main draw path, and is created on first use so projects that never
//! draw captions pay nothing for it.
//!
//! Source pixels are straight (non-premultiplied) RGBA8 — exactly what
//! [`CaptionOverlay::rgba`](crate::captions::CaptionOverlay::rgba) produces —
//! blended `SrcAlpha, OneMinusSrcAlpha`.

use std::borrow::Cow;

/// Full-screen textured quad drawn from three vertices, no vertex buffer.
const OVERLAY_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One oversized triangle covers the viewport with no index or vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // Overlay rows run top-first, so flip v against clip space.
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var overlay_tex: texture_2d<f32>;
@group(0) @binding(1) var overlay_sampler: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(overlay_tex, overlay_sampler, in.uv);
}
"#;

/// GPU resources for the overlay pass, sized to the current overlay buffer.
pub(crate) struct CaptionPass {
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// One pipeline per output format — a target may be sRGB or HDR.
    pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    texture: Option<(u32, u32, wgpu::Texture, wgpu::BindGroup)>,
}

impl CaptionPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers caption overlay"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OVERLAY_WGSL)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers caption overlay bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // The overlay is rasterized at target resolution, so linear filtering
        // only softens the sub-pixel case where it is scaled.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers caption overlay sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            shader,
            layout,
            sampler,
            pipelines: Vec::new(),
            texture: None,
        }
    }

    /// The pipeline for `format`, building it on first sight of that format.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        if let Some(i) = self.pipelines.iter().position(|(f, _)| *f == format) {
            return &self.pipelines[i].1;
        }
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers caption overlay layout"),
            bind_group_layouts: &[Some(&self.layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers caption overlay pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Straight-alpha source-over: the caption sits on top of
                    // whatever the scene already drew.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // No depth: the overlay is 2D and always on top.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.push((format, pipeline));
        &self.pipelines.last().expect("just pushed").1
    }

    /// Upload `rgba` (tightly packed, `width * height * 4`) into the overlay
    /// texture, reallocating when the size changes.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        let stale = !matches!(&self.texture, Some((w, h, _, _)) if *w == width && *h == height);
        if stale {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers caption overlay texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Unorm, not Srgb: the caption colors are authored in the same
                // space the framebuffer encodes, so no extra conversion.
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers caption overlay bind group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.texture = Some((width, height, texture, bind_group));
        }

        let (_, _, texture, _) = self.texture.as_ref().expect("texture just ensured");
        // `write_texture` wants rows padded to 256 bytes.
        const ALIGN: u32 = 256;
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(ALIGN) * ALIGN;
        let staging = if padded == unpadded {
            Cow::Borrowed(rgba)
        } else {
            let mut buf = vec![0u8; (padded * height) as usize];
            for row in 0..height as usize {
                let src = row * unpadded as usize;
                let dst = row * padded as usize;
                buf[dst..dst + unpadded as usize]
                    .copy_from_slice(&rgba[src..src + unpadded as usize]);
            }
            Cow::Owned(buf)
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &staging,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload and blend the overlay onto `target`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        if width == 0 || height == 0 || rgba.len() < (width as usize) * (height as usize) * 4 {
            return;
        }
        self.upload(device, queue, rgba, width, height);
        // `pipeline` may push to `self.pipelines`, so resolve it before the
        // bind group borrow.
        let pipeline_index = match self.pipelines.iter().position(|(f, _)| *f == format) {
            Some(i) => i,
            None => {
                self.pipeline(device, format);
                self.pipelines.len() - 1
            }
        };
        let (_, _, _, bind_group) = self.texture.as_ref().expect("texture uploaded");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers caption overlay encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers caption overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Keep the rendered scene; only blend the caption in.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines[pipeline_index].1);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
