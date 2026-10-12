use std::sync::Arc;

use super::passes::Pass;
use crate::renderer::RenderTarget;

/// Fullscreen vertex shader: emits a single oversized triangle covering NDC
/// and forwards a UV in `[0,1]^2`. Each pass's WGSL pairs with this VS.
///
/// Deliberately *not* [`crate::renderer::fullscreen::VS`], which every other
/// fullscreen pass in the crate now shares. This one puts uv (0,0) at the
/// bottom left — GL's convention, not WebGPU's — and because the chain both
/// reads and writes through it, the flip cancels and the final image is
/// right. Switching would be invisible for a pointwise pass and would move
/// `GlitchPass`'s bands and `FilmPass`'s scanlines, for no gain. A shared
/// helper is worth having; a shared helper that quietly changes two passes'
/// output is not.
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

/// Bytes in a [`PassParams`](super::passes::PassParams) block.
const PASS_PARAMS_SIZE: u64 = 16 * 4;

struct CompiledPass {
    name: &'static str,
    /// Index into `passes`, so the parameters can be re-read each frame.
    index: usize,
    /// This pass's parameter block. One per pass rather than one shared: the
    /// chain is several submits, and `write_buffer` only orders against a
    /// submit, so two passes sharing a buffer would both see the last write.
    params: wgpu::Buffer,
    /// A shader pass carries a pipeline; a [`DrawPass`](super::DrawPass) is
    /// run by index into `passes` instead.
    kind: CompiledKind,
}

enum CompiledKind {
    Shader(wgpu::RenderPipeline),
    /// Index into `passes`, so the pass itself can be called at draw time.
    Draw(usize),
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
    /// The final copy reads no parameters, but the layout has the slot.
    copy_params: Option<wgpu::Buffer>,
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
                // The pass's own parameters. Always present, even for a pass
                // with none: one layout serves every pipeline, so the slot has
                // to be filled either way — a pass without parameters gets a
                // block of zeroes it never reads.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PASS_PARAMS_SIZE),
                    },
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
            copy_params: None,
        }
    }

    pub fn add_pass(&mut self, p: impl Pass + 'static) -> &mut Self {
        self.add_boxed(Box::new(p))
    }

    /// [`add_pass`](Self::add_pass) for a pass that is already boxed — a chain
    /// built at runtime from a list, rather than written out.
    pub fn add_boxed(&mut self, p: Box<dyn Pass>) -> &mut Self {
        self.passes.push(p);
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

    fn params_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: PASS_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn build_pipeline(
        device: &wgpu::Device,
        pass_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        frag_src: &str,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let full_src = format!(
            "{}\n{}\n{}",
            FULLSCREEN_VS,
            super::passes::PASS_PARAMS_WGSL,
            frag_src
        );
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
        if self.copy_params.is_none() {
            self.copy_params = Some(Self::params_buffer(&self.device, "composer copy params"));
        }
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
        for (i, p) in self.passes.iter().enumerate() {
            if p.is_render() {
                continue;
            }
            // A self-drawing pass has no shader to compile; it keeps its place
            // in the chain by index.
            if p.is_draw() {
                self.compiled.push(CompiledPass {
                    name: p.name(),
                    index: i,
                    params: Self::params_buffer(&self.device, p.name()),
                    kind: CompiledKind::Draw(i),
                });
                continue;
            }
            let Some(frag) = p.shader() else {
                continue;
            };
            let pipeline =
                Self::build_pipeline(&self.device, &self.pass_bgl, self.format, frag, p.name());
            self.compiled.push(CompiledPass {
                name: p.name(),
                index: i,
                params: Self::params_buffer(&self.device, p.name()),
                kind: CompiledKind::Shader(pipeline),
            });
        }
        self.compiled_dirty = false;
    }

    fn make_bind_group(
        &self,
        src: &RenderTarget,
        params: &wgpu::Buffer,
        label: &'static str,
    ) -> wgpu::BindGroup {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
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

        // A perspective matrix carries both of these: `[5]` is
        // `1 / tan(fov_y / 2)` and `[0]` is that again over the aspect. Taking
        // them from the matrix rather than from the camera type means an
        // orthographic or a custom camera needs no special case — it just
        // yields values a depth pass has no use for, which is correct, since
        // there is no perspective to reconstruct.
        let proj = camera.projection_matrix().elements;
        let tan_half_fov = if proj[5].abs() > 1e-8 {
            1.0 / proj[5]
        } else {
            1.0
        };
        let aspect = if proj[0].abs() > 1e-8 {
            proj[5] / proj[0]
        } else {
            1.0
        };

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
            // A self-drawing pass gets the two views and the encoder and does
            // its own work; everything else is the fullscreen blit below.
            if let CompiledKind::Draw(index) = cp.kind {
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(cp.name),
                    });
                let mut ctx = super::passes::DrawContext {
                    device: &self.device,
                    queue: &self.queue,
                    // Always `rt_a`'s: that is the one the scene was rendered
                    // into, whatever the colour chain has ping-ponged to.
                    depth: Some(&self.rt_a.depth_texture),
                    near_far: camera.near_far(),
                    tan_half_fov,
                    aspect,
                    encoder: &mut enc,
                    input: &src.color_view,
                    input_texture: &src.color_texture,
                    target: &dst.color_view,
                    target_texture: &dst.color_texture,
                    format: self.format,
                    width: self.width,
                    height: self.height,
                };
                self.passes[index].draw(&mut ctx);
                self.queue.submit(Some(enc.finish()));
                src_is_a = !src_is_a;
                continue;
            }
            let CompiledKind::Shader(pipeline) = &cp.kind else {
                unreachable!("a compiled pass is one kind or the other")
            };
            // Re-read every frame rather than at compile time: a parameter
            // that changes — `FilmPass::time` — has to reach the shader
            // without anything being rebuilt.
            if let Some(params) = self.passes[cp.index].uniforms() {
                self.queue
                    .write_buffer(&cp.params, 0, bytemuck::cast_slice(&params));
            }
            let bg = self.make_bind_group(src, &cp.params, cp.name);
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit(Some(enc.finish()));
            src_is_a = !src_is_a;
        }

        // 3) Copy the latest source into the final target.
        let final_src = if src_is_a { &self.rt_a } else { &self.rt_b };
        let copy_params = self.copy_params.as_ref().expect("copy params");
        let bg = self.make_bind_group(final_src, copy_params, "composer final copy");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocessing::{
        BloomPass, DrawPass, FilmPass, OutlinePass, RenderPass, SsaoPass, SsrPass, ToneMappingPass,
    };
    use crate::renderer::{fullscreen, readback};

    /// A pass that draws itself runs in chain order and its output reaches the
    /// final target.
    ///
    /// The pass here paints a flat colour nothing else in the chain could
    /// produce — the scene is empty and the background is black — so a frame
    /// carrying that colour can only have come from the pass having been
    /// called, with a target the composer went on to read.
    #[test]
    fn a_draw_pass_runs_in_the_chain() {
        let Some(gpu) = crate::gpu::testing::context("composer draw pass") else {
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (64u32, 64u32);

        let mut renderer =
            crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), format, w, h);
        let mut composer = EffectComposer::new(gpu.device.clone(), gpu.queue.clone(), w, h, format);

        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = calls.clone();
        composer.add_pass(RenderPass);
        composer.add_pass(DrawPass::new("flat", move |ctx| {
            seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            assert_eq!((ctx.width, ctx.height), (64, 64));
            assert_eq!(ctx.format, wgpu::TextureFormat::Rgba8Unorm);
            // Clearing is the simplest thing a self-drawing pass can do, and
            // it exercises the part that matters: the view it was handed is
            // the one the chain reads next.
            let _ = fullscreen::begin_with(
                ctx.encoder,
                ctx.target,
                fullscreen::Load::Clear,
                wgpu::Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
                "flat",
            );
        }));

        let out = crate::renderer::RenderTarget::new(&gpu.device, w, h, format);
        let mut scene = crate::Scene::new();
        scene.background = crate::Color::BLACK;
        let mut camera = crate::PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
        camera.position = crate::Vector3::new(0.0, 0.0, 3.0);
        composer.render(&mut renderer, &mut scene, &camera, &out.color_view);

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the draw pass was not called exactly once"
        );
        let pixels = readback::texture_bytes(&gpu.device, &gpu.queue, &out.color_texture);
        let middle = ((h / 2 * w + w / 2) * 4) as usize;
        let (r, g, b) = (pixels[middle], pixels[middle + 1], pixels[middle + 2]);
        assert!(
            g > 200 && r < 40 && b < 40,
            "the final frame is ({r}, {g}, {b}), not the pass's green"
        );
    }

    /// Ambient occlusion darkens the scene, and its strength knob moves it.
    ///
    /// `SsaoPass` was a full-screen copy with a comment saying the real one
    /// lived in JavaScript — from Rust you added ambient occlusion and got a
    /// wasted pass. So the assertion is not that it runs but that the frame is
    /// *darker where geometry meets geometry*, which a copy cannot be.
    #[test]
    fn ambient_occlusion_darkens_the_scene() {
        let Some(gpu) = crate::gpu::testing::context("ssao pass") else {
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (96u32, 96u32);
        let mut renderer =
            crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), format, w, h);

        // A box standing on a floor: the contact between them is the
        // canonical thing ambient occlusion finds, and it covers enough of the
        // frame to measure. Ambient light only, so *nothing else* can produce
        // a gradient — any darkening in the result is the pass's.
        let scene = || {
            let mut scene = crate::Scene::new();
            scene.background = crate::Color::BLACK;
            scene.add_light(crate::AmbientLight::new(crate::Color::WHITE, 1.0));
            let mut floor = crate::Object3D::mesh(crate::Mesh::new(
                crate::BoxGeometry::new(6.0, 0.2, 6.0),
                crate::StandardMaterial::new(crate::Color::WHITE).into(),
            ));
            floor.position = crate::Vector3::new(0.0, -0.6, 0.0);
            scene.add(floor);
            let cube = crate::Object3D::mesh(crate::Mesh::new(
                crate::BoxGeometry::new(1.0, 1.0, 1.0),
                crate::StandardMaterial::new(crate::Color::WHITE).into(),
            ));
            scene.add(cube);
            scene
        };
        let mut camera = crate::PerspectiveCamera::new(50.0, 1.0, 0.1, 20.0);
        camera.position = crate::Vector3::new(1.6, 1.2, 2.6);
        camera.look_at(crate::Vector3::new(0.0, -0.2, 0.0));

        let mut run = |pass: Option<SsaoPass>| -> Vec<u8> {
            let mut composer =
                EffectComposer::new(gpu.device.clone(), gpu.queue.clone(), w, h, format);
            composer.add_pass(RenderPass);
            if let Some(p) = pass {
                composer.add_pass(p);
            }
            let out = crate::renderer::RenderTarget::new(&gpu.device, w, h, format);
            composer.render(&mut renderer, &mut scene(), &camera, &out.color_view);
            readback::texture_bytes(&gpu.device, &gpu.queue, &out.color_texture)
        };

        let plain = run(None);
        // Radius is in *world units*, so it has to suit the scene: this one is
        // about a metre across, and 1.5 reaches across the contact.
        let occluded = run(Some(SsaoPass::new(3.0, 1.5)));
        assert_eq!(plain.len(), occluded.len());

        // Occlusion only ever darkens — a pass that brightened anything would
        // be doing something other than what it says.
        let brightened = plain
            .iter()
            .zip(&occluded)
            .filter(|(a, b)| b.saturating_sub(**a) > 2)
            .count();
        assert_eq!(brightened, 0, "{brightened} channels got *brighter*");

        // And it darkens a real part of the frame, not a rounding error.
        let darkened = |px: &[u8]| -> usize {
            plain
                .chunks_exact(4)
                .zip(px.chunks_exact(4))
                .filter(|(a, b)| a[0] > 24 && (b[0] as f32) < a[0] as f32 * 0.9)
                .count()
        };
        let strong = darkened(&occluded);
        assert!(
            strong > 100,
            "only {strong} pixels of {} were darkened by a tenth — the pass is \
             a copy again",
            plain.len() / 4
        );

        // And the strength knob moves it, so it is not a fixed darkening.
        let gentle = darkened(&run(Some(SsaoPass::new(1.0, 1.5))));
        assert!(
            gentle < strong,
            "strength 1.0 darkened {gentle} pixels against 3.0's {strong}"
        );
    }

    /// A pass with nothing to work with hands the colour on rather than
    /// blacking the frame, and `SsrPass` declines rather than wasting a copy.
    #[test]
    fn ssr_declines_instead_of_pretending() {
        assert!(
            SsrPass.shader().is_none(),
            "SsrPass is advertising a shader again; it has no normals to work \
             from and a silent copy is what this replaced"
        );
        // The composer therefore compiles nothing for it and spends no pass.
        let Some(gpu) = crate::gpu::testing::context("ssr pass") else {
            return;
        };
        let mut composer = EffectComposer::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            32,
            32,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        composer.add_pass(RenderPass);
        composer.add_pass(SsrPass);
        composer.ensure_compiled();
        assert!(
            composer.compiled.is_empty(),
            "the composer built {} pass(es) for a chain that is a render and a \
             decline",
            composer.compiled.len()
        );
    }

    /// Every parameter on every pass, checked by moving it and looking at the
    /// pixels.
    ///
    /// These twelve fields were public, documented by their names, and read by
    /// nothing: the composer's bind-group layout had two entries, texture and
    /// sampler, and each shader hardcoded its own constants. Setting one
    /// compiled, ran, and changed nothing — which is worse than the field not
    /// existing, because the caller believes the picture responded. So the
    /// test is not that the plumbing exists but that each knob *moves the
    /// frame*, one at a time.
    #[test]
    fn every_pass_parameter_reaches_its_shader() {
        let Some(gpu) = crate::gpu::testing::context("pass parameters") else {
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (64u32, 64u32);

        // One frame through a one-pass chain, over a scene chosen so that
        // *every* knob has something to act on: a sphere, so the shading is a
        // smooth gradient rather than flat faces — a flat white box saturates
        // the edge detector, and `edge_strength` of 3 and of 40 then clamp to
        // the same line. Mid-grey and a key light give small luma gradients
        // where a strength actually changes the answer.
        // One renderer for all twenty-four frames: `Renderer::new` compiles
        // every material pipeline, and paying that per case took the test
        // from ten seconds to sixty.
        let mut renderer =
            crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), format, w, h);
        let mut render = |pass: Box<dyn Pass>| -> Vec<u8> {
            let mut composer =
                EffectComposer::new(gpu.device.clone(), gpu.queue.clone(), w, h, format);
            composer.add_pass(RenderPass);
            composer.add_boxed(pass);

            let mut scene = crate::Scene::new();
            scene.background = crate::Color::BLACK;
            scene.add_light(crate::AmbientLight::new(crate::Color::WHITE, 0.25));
            scene.add_light(
                crate::DirectionalLight::new(crate::Color::WHITE, 2.5)
                    .with_direction(crate::Vector3::new(-0.5, -0.7, -0.5).normalize()),
            );
            scene.add(crate::Object3D::mesh(crate::Mesh::new(
                crate::SphereGeometry::new(0.9, 48, 24),
                crate::StandardMaterial::new(crate::Color::from_hex(0x9aa0a6)).into(),
            )));
            let mut camera = crate::PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
            camera.position = crate::Vector3::new(0.9, 0.9, 2.4);
            camera.look_at(crate::Vector3::ZERO);

            let out = crate::renderer::RenderTarget::new(&gpu.device, w, h, format);
            composer.render(&mut renderer, &mut scene, &camera, &out.color_view);
            crate::renderer::readback::texture_bytes(&gpu.device, &gpu.queue, &out.color_texture)
        };

        let differ = |a: &[u8], b: &[u8]| -> usize {
            a.iter().zip(b).filter(|(x, y)| x.abs_diff(**y) > 2).count()
        };

        // Each entry: a baseline pass and the same pass with one field moved.
        // Nothing else changes, so a difference can only be that field.
        type Case = (&'static str, Box<dyn Pass>, Box<dyn Pass>);
        let cases: Vec<Case> = vec![
            (
                "BloomPass::strength",
                Box::new(BloomPass::default()),
                Box::new(BloomPass {
                    strength: 4.0,
                    ..BloomPass::default()
                }),
            ),
            (
                "BloomPass::threshold",
                Box::new(BloomPass::default()),
                Box::new(BloomPass {
                    threshold: 0.05,
                    ..BloomPass::default()
                }),
            ),
            (
                "BloomPass::radius",
                Box::new(BloomPass::default()),
                Box::new(BloomPass {
                    radius: 0.2,
                    ..BloomPass::default()
                }),
            ),
            (
                "OutlinePass::edge_strength",
                Box::new(OutlinePass::default()),
                Box::new(OutlinePass {
                    edge_strength: 40.0,
                    ..OutlinePass::default()
                }),
            ),
            (
                "OutlinePass::edge_thickness",
                Box::new(OutlinePass::default()),
                Box::new(OutlinePass {
                    edge_thickness: 6.0,
                    ..OutlinePass::default()
                }),
            ),
            (
                "OutlinePass::edge_glow",
                Box::new(OutlinePass::default()),
                Box::new(OutlinePass {
                    edge_glow: 8.0,
                    ..OutlinePass::default()
                }),
            ),
            (
                "OutlinePass::edge_color",
                Box::new(OutlinePass::default()),
                Box::new(OutlinePass {
                    edge_color: crate::Color::new(1.0, 0.0, 0.0),
                    ..OutlinePass::default()
                }),
            ),
            (
                "ToneMappingPass::exposure",
                Box::new(ToneMappingPass::default()),
                Box::new(ToneMappingPass { exposure: 0.15 }),
            ),
            (
                "FilmPass::grain_intensity",
                Box::new(FilmPass::default()),
                Box::new(FilmPass {
                    grain_intensity: 0.9,
                    ..FilmPass::default()
                }),
            ),
            (
                "FilmPass::scanline_intensity",
                Box::new(FilmPass::default()),
                Box::new(FilmPass {
                    scanline_intensity: 0.8,
                    ..FilmPass::default()
                }),
            ),
            (
                "FilmPass::scanline_count",
                Box::new(FilmPass::default()),
                Box::new(FilmPass {
                    scanline_count: 11.0,
                    ..FilmPass::default()
                }),
            ),
            (
                "FilmPass::time",
                Box::new(FilmPass::default()),
                Box::new(FilmPass {
                    time: 37.5,
                    ..FilmPass::default()
                }),
            ),
        ];

        for (name, baseline, moved) in cases {
            let before = render(baseline);
            let after = render(moved);
            let n = differ(&before, &after);
            assert!(
                n > 64,
                "{name} changed only {n} of {} bytes — the knob is not reaching the shader",
                before.len()
            );
        }
    }

    /// A chain of shader passes and draw passes keeps its order. The draw pass
    /// writes green; a later shader pass inverts it; red is what comes out.
    #[test]
    fn draw_and_shader_passes_compose_in_order() {
        let Some(gpu) = crate::gpu::testing::context("composer ordering") else {
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (w, h) = (32u32, 32u32);
        let mut renderer =
            crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), format, w, h);
        let mut composer = EffectComposer::new(gpu.device.clone(), gpu.queue.clone(), w, h, format);

        #[derive(Debug)]
        struct Invert;
        impl Pass for Invert {
            fn name(&self) -> &'static str {
                "invert"
            }
            fn shader(&self) -> Option<&'static str> {
                Some(
                    r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(prev_tex, prev_sampler, in.uv);
    return vec4<f32>(vec3<f32>(1.0) - c.rgb, 1.0);
}
"#,
                )
            }
        }

        composer.add_pass(RenderPass);
        composer.add_pass(DrawPass::new("green", move |ctx| {
            let _ = fullscreen::begin_with(
                ctx.encoder,
                ctx.target,
                fullscreen::Load::Clear,
                wgpu::Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
                "green",
            );
        }));
        composer.add_pass(Invert);

        let out = crate::renderer::RenderTarget::new(&gpu.device, w, h, format);
        let mut scene = crate::Scene::new();
        let mut camera = crate::PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
        camera.position = crate::Vector3::new(0.0, 0.0, 3.0);
        composer.render(&mut renderer, &mut scene, &camera, &out.color_view);

        let pixels = readback::texture_bytes(&gpu.device, &gpu.queue, &out.color_texture);
        let middle = ((h / 2 * w + w / 2) * 4) as usize;
        let (r, g, b) = (pixels[middle], pixels[middle + 1], pixels[middle + 2]);
        assert!(
            r > 200 && g < 40 && b > 200,
            "inverting the pass's green should give magenta, got ({r}, {g}, {b})"
        );
    }
}
