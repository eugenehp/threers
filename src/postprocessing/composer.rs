use std::sync::Arc;

use super::passes::Pass;
use crate::renderer::RenderTarget;

/// Fullscreen vertex shader: emits a single oversized triangle covering NDC
/// and forwards a UV in `[0,1]^2`. Each pass's WGSL pairs with this VS.
const FULLSCREEN_VS: &str = r#"
struct VsOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv         : vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> VsOut {
    var out : VsOut;
    let x = f32(i32(i) & 1);
    let y = f32((i32(i) >> 1) & 1);
    out.uv = vec2<f32>(x * 2.0, 1.0 - y * 2.0);
    out.clip = vec4<f32>(x * 4.0 - 1.0, 1.0 - y * 4.0, 0.0, 1.0);
    return out;
}
"#;

struct CompiledPass {
    name: &'static str,
    pipeline: wgpu::RenderPipeline,
}

/// Multi-pass effect chain. Build via `add_pass`, then call
/// `render(renderer, scene, camera, final_target_view)` each frame.
pub struct EffectComposer {
    pub passes: Vec<Box<dyn Pass>>,
    pub render_to_screen: bool,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,

    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    rt_a: RenderTarget,
    rt_b: RenderTarget,

    sampler: wgpu::Sampler,
    pass_bgl: wgpu::BindGroupLayout,
    compiled: Vec<CompiledPass>,
    compiled_dirty: bool,
    copy_pipeline: Option<wgpu::RenderPipeline>,
}

impl EffectComposer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let rt_a = RenderTarget::new(&device, width, height, format);
        let rt_b = RenderTarget::new(&device, width, height, format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers composer sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers composer pass bgl"),
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
        Self {
            passes: Vec::new(),
            render_to_screen: true,
            width,
            height,
            format,
            device,
            queue,
            rt_a,
            rt_b,
            sampler,
            pass_bgl,
            compiled: Vec::new(),
            compiled_dirty: true,
            copy_pipeline: None,
        }
    }

    pub fn add_pass(&mut self, p: impl Pass + 'static) -> &mut Self {
        self.passes.push(Box::new(p));
        self.compiled_dirty = true;
        self
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name()).collect()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.rt_a = RenderTarget::new(&self.device, width, height, self.format);
        self.rt_b = RenderTarget::new(&self.device, width, height, self.format);
    }

    fn build_pipeline(
        device: &wgpu::Device,
        pass_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        frag_src: &str,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let full_src = format!("{}\n{}", FULLSCREEN_VS, frag_src);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(full_src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(pass_bgl)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn ensure_compiled(&mut self) {
        if self.copy_pipeline.is_none() {
            self.copy_pipeline = Some(Self::build_pipeline(
                &self.device,
                &self.pass_bgl,
                self.format,
                super::passes::COPY_FRAG,
                "threers composer copy",
            ));
        }
        if !self.compiled_dirty {
            return;
        }
        self.compiled.clear();
        for p in &self.passes {
            if p.is_render() {
                continue;
            }
            let Some(frag) = p.shader() else {
                continue;
            };
            let pipeline =
                Self::build_pipeline(&self.device, &self.pass_bgl, self.format, frag, p.name());
            self.compiled.push(CompiledPass {
                name: p.name(),
                pipeline,
            });
        }
        self.compiled_dirty = false;
    }

    fn make_bind_group(&self, src: &RenderTarget, label: &'static str) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.pass_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Render the scene through the pass chain. `final_target_view` is the
    /// surface (or another RT) that receives the final composited image.
    pub fn render(
        &mut self,
        renderer: &mut crate::renderer::Renderer,
        scene: &mut crate::scene::Scene,
        camera: &dyn crate::cameras::Camera,
        final_target_view: &wgpu::TextureView,
    ) {
        self.ensure_compiled();

        // 1) Render scene into RT A.
        renderer.render_to(scene, camera, &self.rt_a);

        // 2) Ping-pong fullscreen passes between A and B.
        let mut src_is_a = true;
        for cp in &self.compiled {
            let (src, dst) = if src_is_a {
                (&self.rt_a, &self.rt_b)
            } else {
                (&self.rt_b, &self.rt_a)
            };
            let bg = self.make_bind_group(src, cp.name);
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(cp.name),
                });
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(cp.name),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dst.color_view,
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
                pass.set_pipeline(&cp.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit(Some(enc.finish()));
            src_is_a = !src_is_a;
        }

        // 3) Copy the latest source into the final target.
        let final_src = if src_is_a { &self.rt_a } else { &self.rt_b };
        let bg = self.make_bind_group(final_src, "composer final copy");
        let copy_pipeline = self.copy_pipeline.as_ref().expect("copy pipeline");
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composer final"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composer final"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_target_view,
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
            pass.set_pipeline(copy_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(enc.finish()));
    }
}
