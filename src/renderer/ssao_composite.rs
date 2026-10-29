//! Screen-space ambient occlusion on a live surface.
//!
//! [`Ssao`](crate::renderer::ssao::Ssao) produces an occlusion factor from depth, but only the headless
//! path composites it (into the readback, through `RgbPack`). A canvas has no
//! readback, so this does the same job for one:
//!
//! 1. the scene is drawn into a *presenting* [`RenderTarget`] — tone-mapped
//!    and encoded exactly as the surface would be — with
//!    [`Renderer::render_to`],
//! 2. [`Ssao`](crate::renderer::ssao::Ssao) reads that target's depth into a half-resolution factor,
//! 3. a fullscreen pass blurs it, maps it through a soft response, multiplies
//!    the colour by it in linear light and writes the surface.
//!
//! ```ignore
//! let ao = SsaoComposite::new(&device, surface_format, w, h, 12.0, 6.0).unwrap();
//! ao.render(&mut renderer, &mut scene, &camera, &surface_view);
//! ```
//!
//! THE RESPONSE IS NOT `Ssao`'s. Its `1 - strength * occ` is linear and
//! clamped: on a folded surface the open crests occlude only a few percent, so
//! they need a strength near ten before any shading shows — and at that
//! strength every deep crease and every seam between two meshes clamps to
//! black, a hard-edged streak rather than a shadow. The pass therefore runs
//! `Ssao` at strength 1 (the raw occluded fraction), averages a 3×3 block of
//! it (twelve fixed taps alias into streaks of their own), and applies
//! `1 / (1 + strength * occ)`: steep for a small `occ`, and never zero.

use std::sync::Arc;

use super::ssao::Ssao;
use super::RenderTarget;
use crate::renderer::Renderer;

const COMPOSITE: &str = r#"
struct Params { encoded : u32, strength : f32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var ao : texture_2d<f32>;
@group(0) @binding(2) var<uniform> p : Params;

fn to_linear(c : f32) -> f32 {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}
fn to_srgb(c : f32) -> f32 {
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

@fragment
fn fs_main(in : FullscreenVertex) -> @location(0) vec4<f32> {
    let xy = vec2<u32>(in.clip.xy);
    let c = textureLoad(src, vec2<i32>(xy), 0);
    let sd = textureDimensions(src);
    let ad = textureDimensions(ao);
    // R32Float is not filterable: nearest texel of the half-resolution factor
    let au = vec2<i32>(min(xy * ad / max(sd, vec2<u32>(1u)), ad - vec2<u32>(1u)));
    // A 5x5 box blur over the half-resolution AO (wider than 3x3) smooths the
    // rotated SSAO kernel's residual noise so it reads clean zoomed deep into a
    // fold, without a separate denoise pass.
    var occ = 0.0;
    for (var j = -2; j <= 2; j = j + 1) {
        for (var i = -2; i <= 2; i = i + 1) {
            let q = clamp(au + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(ad) - 1);
            occ = occ + (1.0 - textureLoad(ao, q, 0).r);
        }
    }
    let k = 1.0 / (1.0 + p.strength * occ / 25.0);
    // Occlusion attenuates light, so it multiplies LINEAR values. An sRGB
    // view decodes on load and encodes on store; a plain one holds the
    // encoded bytes, which are decoded and re-encoded here.
    if (p.encoded == 1u) {
        let lin = vec3<f32>(to_linear(c.r), to_linear(c.g), to_linear(c.b)) * k;
        return vec4<f32>(to_srgb(lin.r), to_srgb(lin.g), to_srgb(lin.b), 1.0);
    }
    // Opaque: this is the finished frame. The target's alpha is whatever its
    // blended draws left (a translucent shell over the clear leaves it well
    // under 1), and passing that on darkens everything a later pass blends
    // or a compositor premultiplies — FXAA's included.
    return vec4<f32>(c.rgb * k, 1.0);
}
"#;

/// Ambient occlusion composited onto a surface. See the [module docs](self).
pub struct SsaoComposite {
    target: RenderTarget,
    ssao: Ssao,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    encoded: u32,
    radius: f32,
    strength: f32,
    format: wgpu::TextureFormat,
    /// FXAA's input when it is on (`set_fxaa`): the composite lands here and
    /// FXAA writes the surface.
    fxaa: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl SsaoComposite {
    /// `format` is the surface's; `radius` is in world units and `strength`
    /// scales the response (see the module docs — 4 to 8 suits most scenes).
    /// Sized to one surface: rebuild it when the surface resizes.
    ///
    /// `None` where the device cannot run [`Ssao`]; drawing without occlusion
    /// is then the caller's fallback.
    pub fn new(
        device: &Arc<wgpu::Device>,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        radius: f32,
        strength: f32,
    ) -> Option<Self> {
        let target = RenderTarget::new(device, width, height, format);
        target.set_presents(true);
        let aspect = target.width as f32 / target.height.max(1) as f32;
        // Placeholder camera numbers: `render`/`finish` refresh them from the
        // renderer every frame, before anything is recorded.
        let ssao = Ssao::new(device, &target.depth_texture, 0.1, 1000.0, radius, 1.0, 0.5, aspect)?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao composite"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", super::fullscreen::VS, COMPOSITE).into()),
        });
        let texture = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao composite"),
            entries: &[
                texture(0, true),
                texture(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssao composite"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao composite"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(super::fullscreen::VS_ENTRY),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let encoded = u32::from(!format.is_srgb());
        let params = wgpu::util::DeviceExt::create_buffer_init(
            &**device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("ssao composite params"),
                contents: bytemuck::cast_slice(&[encoded, strength.to_bits(), 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao composite"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ssao.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        Some(Self {
            target,
            ssao,
            pipeline,
            bind_group,
            params,
            encoded,
            radius,
            strength,
            format,
            fxaa: None,
        })
    }

    /// The offscreen target the scene is drawn into before [`finish`](Self::finish).
    pub fn target(&self) -> &RenderTarget {
        &self.target
    }

    /// Change the occlusion's world radius and response without a rebuild.
    pub fn set_params(&mut self, queue: &wgpu::Queue, radius: f32, strength: f32) {
        self.radius = radius;
        self.strength = strength;
        queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[self.encoded, strength.to_bits(), 0, 0]));
    }

    /// Antialias the composited frame with FXAA (three.js's `FXAAShader`, the
    /// renderer's own port) before it reaches the surface.
    ///
    /// The scene is drawn into [`target`](Self::target) single-sampled — the
    /// occlusion needs that target's depth, which a multisampled draw does not
    /// write — so without this its silhouettes are as hard-edged as the
    /// pixels. One extra fullscreen pass over a same-sized texture.
    pub fn set_fxaa(&mut self, device: &wgpu::Device, on: bool) {
        if !on {
            self.fxaa = None;
            return;
        }
        if self.fxaa.is_some() {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao composite fxaa input"),
            size: wgpu::Extent3d { width: self.target.width, height: self.target.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.fxaa = Some((tex, view));
    }

    /// Draw `scene` into the target, then occlude and composite onto `view`
    /// (through FXAA when [`set_fxaa`](Self::set_fxaa) is on).
    pub fn render(
        &self,
        renderer: &mut Renderer,
        scene: &mut crate::scene::Scene,
        camera: &dyn crate::cameras::Camera,
        view: &wgpu::TextureView,
    ) {
        renderer.render_to(scene, camera, &self.target);
        match &self.fxaa {
            Some((_, input)) => {
                self.finish(renderer, input);
                let (w, h) = (self.target.width, self.target.height);
                // postfx kind 1 is FXAA; `params2.xy` its texel size
                renderer.apply_postfx(input, view, 1, 0.0, [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0], [0.0; 4], w, h);
            }
            None => self.finish(renderer, view),
        }
    }

    /// Occlude and composite onto `view`, after the scene was drawn into
    /// [`target`](Self::target).
    ///
    /// The camera's planes, lens and depth direction are read back from
    /// `renderer` on every call: occlusion computed with another frame's near
    /// plane is wrong in a way nothing reports.
    pub fn finish(&self, renderer: &Renderer, view: &wgpu::TextureView) {
        let (near, far) = renderer.camera_planes();
        let (tan_half_fov, aspect) = renderer.camera_lens();
        self.composite(&renderer.device_arc(), &renderer.queue_arc(), view, near, far, tan_half_fov, aspect, renderer.reversed_z());
    }

    #[allow(clippy::too_many_arguments)]
    fn composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        near: f32,
        far: f32,
        tan_half_fov: f32,
        aspect: f32,
        reversed: bool,
    ) {
        self.ssao.set_reversed_depth(queue, reversed);
        self.ssao.set_range(queue, near, far, self.radius, 1.0, tan_half_fov, aspect);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ssao composite"),
        });
        self.ssao.record(&mut encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 64;
    const GREY: u8 = 180;

    /// A flat wall must come through untouched in either kind of format — the
    /// decode/encode round trip of a plain one included — while a depth step
    /// darkens the far side of its crease without ever reaching black.
    #[test]
    fn composite_keeps_flat_walls_and_greys_creases() {
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let device = Arc::new(device);
        for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb] {
            let ao = SsaoComposite::new(&device, format, W, W, 2.0, 6.0).expect("ssao composite");

            fill(&device, &queue, &ao, false);
            let flat = run(&device, &queue, &ao, format);
            let off = flat.iter().map(|v| (*v as i32 - GREY as i32).abs()).max().unwrap();
            assert!(off <= 1, "{format:?}: a flat wall moved by {off} — it should not occlude");

            fill(&device, &queue, &ao, true);
            let step = run(&device, &queue, &ao, format);
            let darkest = *step.iter().min().unwrap();
            assert!(darkest < GREY - 10, "{format:?}: the crease is not occluded (darkest {darkest})");
            assert!(darkest > 0, "{format:?}: the crease clamped to black");
        }
    }

    /// Colour `GREY` everywhere, depth either a flat wall or a step across the
    /// middle (the near half in front of the far one).
    fn fill(device: &Arc<wgpu::Device>, queue: &wgpu::Queue, ao: &SsaoComposite, step: bool) {
        let body = if step {
            "if (in.clip.x < 32.0) { o.depth = 0.30; } else { o.depth = 0.80; }"
        } else {
            "o.depth = 0.5;"
        };
        let g = GREY as f32 / 255.0;
        let src = format!(
            "{}\nstruct Out {{ @location(0) c : vec4<f32>, @builtin(frag_depth) depth : f32 }};\n\
             @fragment fn fs_main(in : FullscreenVertex) -> Out {{ var o : Out; o.c = vec4<f32>({g}, {g}, {g}, 1.0); {body} return o; }}",
            super::super::fullscreen::VS
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let t = ao.target();
        // the target may be sRGB; write the bytes through a plain view so
        // GREY lands as stored, the way a presenting pass leaves it
        let plain = t.color_texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(t.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(super::super::fullscreen::VS_ENTRY),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(t.format.remove_srgb_suffix().into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let depth = t.depth_texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &plain,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Composite into a fresh texture of `format` and read its red channel back
    /// as stored bytes.
    fn run(device: &Arc<wgpu::Device>, queue: &wgpu::Queue, ao: &SsaoComposite, format: wgpu::TextureFormat) -> Vec<u8> {
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = out.create_view(&Default::default());
        ao.composite(device, queue, &view, 0.1, 50.0, 0.5, 1.0, false);
        let bytes_per_row = (W * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bytes_per_row * W) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(W),
                },
            },
            wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = readback.slice(..).get_mapped_range().expect("buffer range is mapped");
        (0..W)
            .flat_map(|row| {
                let start = (row * bytes_per_row) as usize;
                data[start..start + (W * 4) as usize].chunks_exact(4).map(|c| c[0]).collect::<Vec<_>>()
            })
            .collect()
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ssao composite test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }
}
