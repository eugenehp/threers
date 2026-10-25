//! Clearing part of a target: a view's rectangle within a canvas holding
//! several (deck.gl's `View` `clear`), where a pass's load-op clear would
//! wipe them all.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use super::{MarkTarget, DEPTH_FORMAT};

const SHADER: &str = r#"
struct Clear {
    color : vec4<f32>,
    // x: the depth written.
    depth : vec4<f32>,
};
@group(0) @binding(0) var<uniform> clear : Clear;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, clear.depth.x, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return clear.color;
}
"#;

/// `(format, samples, has depth, writes colour, writes depth)`.
type Key = (wgpu::TextureFormat, u32, bool, bool, bool);

pub(crate) struct Clearer {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<Key, wgpu::RenderPipeline>,
}

impl Clearer {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers rect clear"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers rect clear bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers rect clear layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        Self { module, layout, pipeline_layout, pipelines: HashMap::new() }
    }

    fn pipeline(&mut self, device: &wgpu::Device, key: Key) -> &wgpu::RenderPipeline {
        let (format, samples, has_depth, color, depth) = (key.0, key.1, key.2, key.3, key.4);
        let (module, layout) = (&self.module, &self.pipeline_layout);
        self.pipelines.entry(key).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers rect clear"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: if color { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() },
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: has_depth.then(|| wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// Set `target`'s rectangle (all of it without one) to `color` — as a
    /// pass's load-op clear takes it — and its depth to `depth`; `None`
    /// leaves either as it is.
    pub(crate) fn clear(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &MarkTarget,
        color: Option<wgpu::Color>,
        depth: Option<f32>,
    ) {
        let depth = depth.filter(|_| target.depth.is_some());
        if color.is_none() && depth.is_none() {
            return;
        }
        let c = color.unwrap_or(wgpu::Color::TRANSPARENT);
        let uniform: [f32; 8] = [c.r as f32, c.g as f32, c.b as f32, c.a as f32, depth.unwrap_or(1.0), 0.0, 0.0, 0.0];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers rect clear values"),
            contents: bytemuck::cast_slice(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers rect clear bg"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        let key = (target.format, target.samples, target.depth.is_some(), color.is_some(), depth.is_some());
        let pipeline = self.pipeline(device, key).clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("threers rect clear") });
        {
            let mut pass = super::MarkRenderer::begin_target_pass(&mut encoder, target);
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
