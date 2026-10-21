//! Post-fx draws described as data.
//!
//! [`Renderer::run_postfx`] is what the three.js-style composer in
//! [`crate::postprocessing`] runs every pass through. It is the same
//! `POSTFX_SHADER` and the same bind-group layout as
//! [`Renderer::apply_postfx_by_id`], with what that entry point fixes per kind
//! made explicit instead:
//!
//! * every texture is read *linear*: a half-float target as it is, an 8-bit
//!   one through its decoding (`…Srgb`) view, so a kind never has to know which
//!   it was given;
//! * the blend is named ([`PostFxBlend`]) rather than inferred from the kind,
//!   because three.js passes draw the same shader both ways — a `CopyShader`
//!   replaces in one place and adds in another;
//! * which target's hardware depth is bound, and what fills the
//!   `noise_tex` slot, are part of the description, so a pass that needs two
//!   depth buffers (`OutlinePass`'s hidden-edge test) can have them.
//!
//! An sRGB output is still handled the one way `POSTFX_SHADER` knows
//! (`postfx_flag` bit 1), which is what keeps WebGL2 — where the canvas and
//! every 8-bit target are sRGB — storing the bytes WebGPU stores.

use std::collections::HashMap;

use super::{rt_sample_format, PostFxCamera, Renderer};

/// How a post-fx draw combines with what its output already holds. The names
/// are three.js's blending modes, and each has its blend equation exactly —
/// alpha included, since a later `AdditiveBlending` copy multiplies by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PostFxBlend {
    /// Clear to [`PostFxOp::clear`], then write: `NoBlending` into a target
    /// the pass clears first.
    Replace,
    /// Write without clearing. Every kind covers the whole target, so this
    /// differs from [`Replace`](Self::Replace) only where a kind outputs
    /// nothing — which none does — and in not paying for the clear.
    Overwrite,
    /// `ONE, ONE` — premultiplied `AdditiveBlending`.
    Add,
    /// `SRC_ALPHA, ONE` on colour and alpha — `AdditiveBlending` with straight
    /// alpha, which UnrealBloomPass, BloomPass and OutlinePass composite with.
    AddAlpha,
    /// `DST_COLOR, ZERO` (alpha `DST_ALPHA, ZERO`) — SSAOPass's and SAOPass's
    /// `CustomBlending`, which multiplies the occlusion into the image.
    Multiply,
    /// `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` (alpha `ONE, ONE_MINUS_SRC_ALPHA`) —
    /// `NormalBlending` with straight alpha.
    Normal,
    /// `ONE, ONE_MINUS_SRC_ALPHA` — `NormalBlending` with `premultipliedAlpha`.
    NormalPremultiplied,
}

impl PostFxBlend {
    fn state(self) -> Option<wgpu::BlendState> {
        use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
        let c = |src, dst| C {
            src_factor: src,
            dst_factor: dst,
            operation: O::Add,
        };
        match self {
            Self::Replace | Self::Overwrite => None,
            Self::Add => Some(wgpu::BlendState {
                color: c(F::One, F::One),
                alpha: c(F::One, F::One),
            }),
            Self::AddAlpha => Some(wgpu::BlendState {
                color: c(F::SrcAlpha, F::One),
                alpha: c(F::SrcAlpha, F::One),
            }),
            Self::Multiply => Some(wgpu::BlendState {
                color: c(F::Dst, F::Zero),
                alpha: c(F::DstAlpha, F::Zero),
            }),
            Self::Normal => Some(wgpu::BlendState {
                color: c(F::SrcAlpha, F::OneMinusSrcAlpha),
                alpha: c(F::One, F::OneMinusSrcAlpha),
            }),
            Self::NormalPremultiplied => Some(wgpu::BlendState {
                color: c(F::One, F::OneMinusSrcAlpha),
                alpha: c(F::One, F::OneMinusSrcAlpha),
            }),
        }
    }
}

/// What fills `POSTFX_SHADER`'s `noise_tex` slot (binding 8, unfilterable,
/// sampled nearest and repeating).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostFxNoise {
    /// The SSAO rotation noise — or, for the glitch kinds, the displacement
    /// map — as [`Renderer::apply_postfx_by_id`] binds it.
    Default,
    /// The hardware depth of a registered render target: a second depth
    /// buffer next to [`PostFxOp::hw_depth`].
    Depth(u32),
    /// The colour of a registered render target, read through a view that
    /// does not decode.
    Color(u32),
}

/// One post-fx draw. Ids name render targets registered with
/// [`Renderer::register_render_target`]; `0` binds a neutral stand-in.
#[derive(Clone, Debug)]
pub struct PostFxOp {
    /// The `POSTFX_SHADER` branch.
    pub kind: u32,
    /// `input_tex`, sampled linear and clamped.
    pub input: u32,
    /// `depth_tex` — any colour target, despite the name.
    pub depth: u32,
    /// `normal_tex` — likewise.
    pub normal: u32,
    /// The target whose depth attachment binds as `hw_depth_tex`; non-zero
    /// also sets `params.w`, which the depth-reading kinds test.
    pub hw_depth: u32,
    pub noise: PostFxNoise,
    /// `params.y`.
    pub time: f32,
    /// `params.z`.
    pub p0z: f32,
    pub p2: [f32; 4],
    pub p3: [f32; 4],
    pub blend: PostFxBlend,
    pub camera: PostFxCamera,
    /// Sample `depth_tex` and `normal_tex` bilinearly rather than nearest.
    pub linear_aux: bool,
    /// The output is the display — the canvas, or a half-float target
    /// standing in for it while several draws are blended — so the kinds
    /// clamp, and convert where three.js converts, rather than storing the
    /// value. A [`PostFxOutput::View`] always is. Otherwise an 8-bit target
    /// is written through its sRGB view, so it reads back linear like a
    /// half-float one.
    pub display: bool,
    /// The clear colour for [`PostFxBlend::Replace`], linear (see
    /// [`Renderer::clear_postfx_output`]).
    pub clear: [f64; 4],
}

impl PostFxOp {
    /// `kind` reading `input`, everything else neutral.
    pub fn new(kind: u32, input: u32) -> Self {
        Self {
            kind,
            input,
            depth: 0,
            normal: 0,
            hw_depth: 0,
            noise: PostFxNoise::Default,
            time: 0.0,
            p0z: 0.0,
            p2: [0.0; 4],
            p3: [0.0; 4],
            blend: PostFxBlend::Replace,
            camera: PostFxCamera::default(),
            linear_aux: false,
            display: false,
            clear: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Where a [`PostFxOp`] writes.
pub enum PostFxOutput<'a> {
    /// A registered render target.
    Target(u32),
    /// A view the caller owns — a swapchain image. `format` must be the
    /// view's, and `width × height` its size.
    View {
        view: &'a wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    },
}

/// Lazily built state for [`Renderer::run_postfx`]: the module and a pipeline
/// per (blend, output format) pair actually used.
#[derive(Default)]
pub(crate) struct PostFxExt {
    module: Option<wgpu::ShaderModule>,
    layout: Option<wgpu::PipelineLayout>,
    pipelines: HashMap<(PostFxBlend, wgpu::TextureFormat), wgpu::RenderPipeline>,
}

/// `POSTFX_SHADER`'s uniform block.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    params: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    resolution: [f32; 4],
    ssao0: [f32; 4],
    inv_proj: [f32; 16],
    proj: [f32; 16],
}

impl Renderer {
    /// Forget a render target registered with
    /// [`register_render_target`](Self::register_render_target): its cached
    /// views go, and the registry's reference with them, so a target the
    /// caller has dropped is freed. A pass reallocating its targets on resize
    /// calls this for the old ones.
    pub fn unregister_render_target(&mut self, rt_id: u32) {
        self.rt_view_cache.remove(&rt_id);
        self.rt_linear_view_cache.remove(&rt_id);
        self.rt_depth_view_cache.remove(&rt_id);
        if self.rt_registry.remove(&rt_id).is_some() {
            self.tex_bg_cache.clear();
        }
    }

    /// The registered target `rt_id`, if there is one.
    pub fn registered_render_target(
        &self,
        rt_id: u32,
    ) -> Option<std::sync::Arc<crate::renderer::RenderTarget>> {
        self.rt_registry.get(&rt_id).cloned()
    }

    /// Fill `POSTFX_SHADER`'s 32-entry `vec4` uniform array — the SSAO kernel
    /// slot, which the blur and composite kinds also read their weights from.
    /// Each draw is its own submit, so a pass may rewrite it between draws.
    pub fn set_postfx_kernel(&mut self, data: &[[f32; 4]]) {
        let mut block = [[0f32; 4]; 32];
        for (dst, src) in block.iter_mut().zip(data) {
            *dst = *src;
        }
        self.queue
            .write_buffer(&self.ssao_kernel_buffer, 0, bytemuck::cast_slice(&block));
    }

    /// Write `texels` (RGBA, `width × height`, row-major from the top) into a
    /// registered half-float target — data a pass samples rather than renders:
    /// a LUT, SMAA's lookup tables, a TexturePass map.
    pub fn upload_render_target_f16(&mut self, rt_id: u32, texels: &[[f32; 4]]) {
        let Some(rt) = self.rt_registry.get(&rt_id).cloned() else {
            return;
        };
        if rt.format != wgpu::TextureFormat::Rgba16Float
            || texels.len() < (rt.width * rt.height) as usize
        {
            return;
        }
        let mut bytes = Vec::with_capacity(texels.len() * 8);
        for t in &texels[..(rt.width * rt.height) as usize] {
            for c in t {
                bytes.extend_from_slice(
                    &crate::renderer::gpu_texture::f32_to_f16_bits(*c).to_le_bytes(),
                );
            }
        }
        let row = rt.width * 8;
        let padded = row.div_ceil(256) * 256;
        let mut upload = vec![0u8; (padded * rt.height) as usize];
        for y in 0..rt.height as usize {
            let src = y * row as usize;
            let dst = y * padded as usize;
            upload[dst..dst + row as usize].copy_from_slice(&bytes[src..src + row as usize]);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &rt.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(rt.height),
            },
            wgpu::Extent3d {
                width: rt.width,
                height: rt.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Clear `output` to the linear colour `color` — three.js
    /// `renderer.clear()` on a target, for the whole of it whatever the
    /// viewport. `display`: the output holds display values (the screen or a
    /// stand-in for it), so the colour lands as its sRGB encoding, as three.js
    /// clears a canvas; otherwise it is stored so it reads back as `color`.
    pub fn clear_postfx_output(
        &mut self,
        output: PostFxOutput<'_>,
        color: [f64; 4],
        display: bool,
    ) {
        let rt;
        let (view, format) = match output {
            PostFxOutput::Target(id) => {
                let Some(t) = self.rt_registry.get(&id).cloned() else {
                    return;
                };
                rt = t;
                (&rt.color_view, rt.format)
            }
            PostFxOutput::View { view, format, .. } => (view, format),
        };
        let c = stored_clear(color, format, display);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers postfx clear"),
            });
        {
            let _pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers postfx clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(c),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn postfx_pipeline(
        &mut self,
        blend: PostFxBlend,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let ext = &mut self.postfx_ext;
        if ext.module.is_none() {
            ext.module = Some(
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("threers postfx op shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            crate::renderer::shader::POSTFX_SHADER.into(),
                        ),
                    }),
            );
        }
        if ext.layout.is_none() {
            ext.layout = Some(self.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("threers postfx op layout"),
                    bind_group_layouts: &[Some(&self.postfx_bgl)],
                    immediate_size: 0,
                },
            ));
        }
        let (module, layout) = (ext.module.as_ref().unwrap(), ext.layout.as_ref().unwrap());
        let device = &self.device;
        ext.pipelines.entry((blend, format)).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx op pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: blend.state(),
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
        })
    }

    /// The colour of a registered target as a kind reads it: linear — a
    /// half-float target as stored, an 8-bit one through its decoding view.
    fn postfx_color_view(&self, id: u32) -> Option<wgpu::TextureView> {
        let rt = self.rt_registry.get(&id)?;
        Some(rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers postfx op color"),
            format: Some(rt_sample_format(rt.format)),
            ..Default::default()
        }))
    }

    fn postfx_depth_view(&self, id: u32) -> Option<wgpu::TextureView> {
        let rt = self.rt_registry.get(&id)?;
        Some(rt.depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers postfx op depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        }))
    }

    /// Run one post-fx draw. See [`PostFxOp`].
    pub fn run_postfx(&mut self, op: &PostFxOp, output: PostFxOutput<'_>) {
        // A swapchain view is the display by definition.
        let display = op.display || matches!(output, PostFxOutput::View { .. });
        let out_rt = match &output {
            PostFxOutput::Target(id) => match self.rt_registry.get(id) {
                Some(rt) => Some(rt.clone()),
                None => return,
            },
            PostFxOutput::View { .. } => None,
        };
        // A stored 8-bit target is written through its sRGB view, so the
        // hardware encodes what the kind writes linear and its decoding view
        // reads it back linear — as a half-float target holds it.
        let srgb_view = out_rt.as_ref().and_then(|rt| {
            let srgb = rt.format.add_srgb_suffix();
            (!display && rt.format != wgpu::TextureFormat::Rgba16Float && srgb != rt.format).then(
                || {
                    rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("threers postfx op srgb output"),
                        format: Some(srgb),
                        ..Default::default()
                    })
                },
            )
        });
        let (out_view, out_format, width, height) = match (&output, &out_rt) {
            (_, Some(rt)) => match &srgb_view {
                Some(v) => (v, rt.format.add_srgb_suffix(), rt.width, rt.height),
                None => (&rt.color_view, rt.format, rt.width, rt.height),
            },
            (
                PostFxOutput::View {
                    view,
                    format,
                    width,
                    height,
                },
                None,
            ) => (*view, *format, *width, *height),
            _ => return,
        };
        // Built (or fetched) before anything below borrows `self`.
        self.postfx_pipeline(op.blend, out_format);
        let fallback = || {
            self.default_normal
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let input = self.postfx_color_view(op.input).unwrap_or_else(fallback);
        let depth = self.postfx_color_view(op.depth).unwrap_or_else(fallback);
        let normal = self.postfx_color_view(op.normal).unwrap_or_else(fallback);
        let hw_depth = self.postfx_depth_view(op.hw_depth).unwrap_or_else(|| {
            self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers postfx op hw depth fallback"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            })
        });
        let noise_owned = match op.noise {
            PostFxNoise::Default => None,
            PostFxNoise::Depth(id) => self.postfx_depth_view(id),
            PostFxNoise::Color(id) => self.rt_registry.get(&id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx op noise"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            }),
        };
        let noise = noise_owned
            .as_ref()
            .unwrap_or(if op.kind == 5 || op.kind == 31 {
                &self.glitch_disp_view
            } else {
                &self.ssao_noise_view
            });
        let linear_out = !display;
        let uniforms = Uniforms {
            params: [
                op.kind as f32,
                op.time,
                op.p0z,
                if op.hw_depth != 0 { 1.0 } else { 0.0 },
            ],
            params2: op.p2,
            params3: op.p3,
            resolution: [
                width as f32,
                height as f32,
                if linear_out { 1.0 } else { 0.0 },
                // Bit 1: the display is sRGB and encodes on write, so the
                // kind's display value is decoded first (`postfx_flag`).
                // Bit 16: the depth buffers were drawn reversed
                // (`set_reversed_z`); the kinds read them back as `1 - d`.
                (u32::from(display && out_format.is_srgb())
                    | if self.state_pipelines.reversed() {
                        16
                    } else {
                        0
                    }) as f32,
            ],
            ssao0: [
                op.camera.near,
                op.camera.far,
                op.camera.kernel_radius,
                op.camera.kernel_size as f32,
            ],
            inv_proj: op.camera.inv_proj,
            proj: op.camera.proj,
        };
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers postfx op uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let aux_sampler = if op.linear_aux || op.kind == 5 || op.kind == 31 {
            &self.postfx_sampler
        } else {
            &self.postfx_nearest_sampler
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers postfx op bg"),
            layout: &self.postfx_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.postfx_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(aux_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&hw_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.ssao_kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(noise),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.postfx_noise_sampler),
                },
            ],
        });
        let load = if op.blend == PostFxBlend::Replace {
            wgpu::LoadOp::Clear(stored_clear(op.clear, out_format, display))
        } else {
            wgpu::LoadOp::Load
        };
        let pipeline = &self.postfx_ext.pipelines[&(op.blend, out_format)];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers postfx op"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers postfx op"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: out_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

/// The value to clear a `format` target to so it holds the linear colour `c`
/// as that target holds colour: a display holds the sRGB encoding (an sRGB
/// format encodes on its own); a stored 8-bit target is read back through a
/// decoding view, so it too holds the encoding; a half-float one, `c`.
fn stored_clear(c: [f64; 4], format: wgpu::TextureFormat, display: bool) -> wgpu::Color {
    let encode = !format.is_srgb() && (display || format != wgpu::TextureFormat::Rgba16Float);
    let e = |v: f64| {
        if encode {
            crate::utils::srgb::linear_to_srgb_f32(v as f32) as f64
        } else {
            v
        }
    };
    wgpu::Color {
        r: e(c[0]),
        g: e(c[1]),
        b: e(c[2]),
        a: c[3],
    }
}
