//! Averaging jittered frames at **output** resolution.
//!
//! The temporal pass in [`Renderer`](super::Renderer) accumulates where it
//! sits, which is before the supersample resolve — so at `supersample(2)` it
//! reads and writes four times more pixels than the frame it is building. Per
//! sample at 1080p that is 166 MB of traffic against 66 MB for the same work
//! done after the resolve, and it holds 133 MB of history buffers instead of
//! 33 MB.
//!
//! Only one of those two numbers turned into anything. The memory did: peak
//! footprint for an 8K render fell from 11.07 GB to 8.78 GB. The traffic did
//! not — 74.3 ms a frame against 76.9, inside a ±6 ms spread — because the
//! supersample resolve, which used to run once a frame, now runs once a sample
//! and hands most of the saving back. Reach for this for the headroom, not for
//! the frame rate.
//!
//! Swapping the order is free *provided both means are taken in the same
//! space*: `mean(resolve(x))` and `resolve(mean(x))` agree because averaging
//! and the resolve are both linear. This runs the resolve first and adds the
//! result into one float buffer with the blender, which also does away with the
//! ping-pong — there is one target, not two, and no bind group that has to
//! alternate with the frame index.
//!
//! The catch is the transfer curve. Every target except `Rgba16Float` holds
//! sRGB-*encoded* bytes, and the resolve already decodes before it averages
//! four subpixels. Summing those encoded bytes would be a mean over a curve —
//! on a black-to-white edge it reads 21/255 where linear light gives 0, because
//! sRGB is convex and Jensen's inequality does the rest. So this decodes on the
//! way in and re-encodes on the way out, and the two means compose exactly.

use wgpu::util::DeviceExt;

/// One accumulation buffer and the two passes that fill and drain it.
pub(crate) struct Accumulator {
    /// Held only to keep `view` alive.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Adds a source texture into the buffer, weighted.
    add: wgpu::RenderPipeline,
    add_layout: wgpu::BindGroupLayout,
    /// Divides the buffer by the accumulated weight, into a caller's target.
    resolve: wgpu::RenderPipeline,
    /// One weight buffer per pass, not one shared.
    ///
    /// `Queue::write_buffer` is ordered against *submits*, not against the
    /// passes inside an encoder: recording add-then-resolve in one encoder with
    /// a single buffer means both passes read whichever value was written last.
    /// That failed quietly and plausibly — a 16-sample mean came out as
    /// `(15 + 1/16)/16`, a 6% darkening that looks like a gamma slip rather
    /// than a bug.
    w_add: wgpu::Buffer,
    w_res: wgpu::Buffer,
    /// Built once: the resolve pass always reads this buffer's own view.
    resolve_bg: wgpu::BindGroup,
    /// The add pass reads whatever the caller hands it, which is the same
    /// texture every sample — so keep the bind group and rebuild it only if the
    /// texture changes. Sixteen samples a frame at 30 fps is a thousand bind
    /// groups a second otherwise, and it showed: the per-sample allocations
    /// cost more than the bandwidth this whole module saves.
    add_bg: std::cell::RefCell<Option<(u64, wgpu::BindGroup)>>,
    /// Format the `resolve` pipeline was built against.
    out_format: wgpu::TextureFormat,
}

impl Accumulator {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        out_format: wgpu::TextureFormat,
    ) -> Self {
        // Whether these bytes arrive on a curve the shader has to undo. An
        // sRGB format decodes in hardware when sampled, so there it does not.
        let encoded = super::shader_applies_transfer(out_format);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers accumulate"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    crate::renderer::fullscreen::VS,
                    shader_source(encoded)
                )
                .into(),
            ),
        });
        let mk_layout = |label| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        };
        let add_layout = mk_layout("threers accumulate add layout");
        let resolve_layout = mk_layout("threers accumulate resolve layout");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers accumulation buffer"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Float, and for the same reason the temporal history is: a running
            // sum of sixteen samples in eight bits stops improving around the
            // eighth, and the caller keeps paying for the rest.
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mk_pipeline = |label, entry, layout: &wgpu::BindGroupLayout, fmt, blend| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(crate::renderer::fullscreen::VS_ENTRY),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // `One, One`: the blender does the running sum, so nothing has to read
        // the buffer back to add to it.
        let additive = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        });
        let add = mk_pipeline(
            "threers accumulate add",
            "fs_add",
            &add_layout,
            wgpu::TextureFormat::Rgba16Float,
            additive,
        );
        let resolve = mk_pipeline(
            "threers accumulate resolve",
            "fs_resolve",
            &resolve_layout,
            out_format,
            Some(wgpu::BlendState::REPLACE),
        );
        let weight_buf = |label| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let w_add = weight_buf("threers accumulate add weight");
        let w_res = weight_buf("threers accumulate resolve weight");
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers accumulate resolve bind"),
            layout: &resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_res.as_entire_binding(),
                },
            ],
        });
        Self {
            _texture: texture,
            view,
            width,
            height,
            add,
            add_layout,
            resolve,
            w_add,
            w_res,
            resolve_bg,
            add_bg: std::cell::RefCell::new(None),
            out_format,
        }
    }

    pub fn matches(&self, width: u32, height: u32, out_format: wgpu::TextureFormat) -> bool {
        self.width == width && self.height == height && self.out_format == out_format
    }

    /// Add `src` into the buffer, scaled by `weight`. `clear` starts a new run.
    ///
    /// `key` identifies `src` so the bind group can be reused between samples;
    /// pass something that changes when the texture does.
    pub fn add_sample(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        key: u64,
        weight: f32,
        clear: bool,
    ) {
        queue.write_buffer(
            &self.w_add,
            0,
            bytemuck::cast_slice(&[weight, 0.0, 0.0, 0.0]),
        );
        let mut cached = self.add_bg.borrow_mut();
        if cached.as_ref().is_none_or(|(k, _)| *k != key) {
            *cached = Some((
                key,
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers accumulate add bind"),
                    layout: &self.add_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.w_add.as_entire_binding(),
                        },
                    ],
                }),
            ));
        }
        let bg = &cached.as_ref().unwrap().1;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers accumulate add pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.add);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Divide by `total` and write into `dst`.
    pub fn resolve_into(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        total: f32,
    ) {
        queue.write_buffer(
            &self.w_res,
            0,
            bytemuck::cast_slice(&[1.0 / total.max(1e-6), 0.0, 0.0, 0.0]),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers accumulate resolve pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, &self.resolve_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn shader_source(encoded: bool) -> String {
    let (decode, encode) = if encoded {
        ("srgb_to_linear(c.rgb)", "linear_to_srgb(c.rgb)")
    } else {
        ("c.rgb", "c.rgb")
    };
    SHADER.replace("DECODE", decode).replace("ENCODE", encode)
}

const SHADER: &str = r#"
struct W { weight : f32, _p0 : f32, _p1 : f32, _p2 : f32 };
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var<uniform> w : W;

@fragment
fn fs_add(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(DECODE, c.a) * w.weight;
}

@fragment
fn fs_resolve(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    var c = textureLoad(src, vec2<i32>(pos.xy), 0) * w.weight;
    // Negative light is filter ringing, not signal, and the encode is undefined
    // there.
    c = vec4<f32>(max(c.rgb, vec3<f32>(0.0)), clamp(c.a, 0.0, 1.0));
    return vec4<f32>(ENCODE, c.a);
}

fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)),
                  c / 12.92,
                  c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055,
                  c * 12.92,
                  c <= vec3<f32>(0.0031308));
}
"#;
