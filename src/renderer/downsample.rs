//! Averaging a supersampled render down to its output size, on the GPU.
//!
//! Supersampling renders at `size × factor` and averages each `factor × factor`
//! block down to one pixel. Doing that on the CPU means reading the *large*
//! frame across the bus and then walking it with a transfer function per
//! channel per sample, and it is far more expensive than anything else in a
//! headless frame — at 1920×1080 with `supersample(2)`:
//!
//! ```text
//! render + readback of 3840×2160   4.4 ms
//! CPU average down to 1920×1080   33.1 ms
//! ```
//!
//! Seven times the cost of drawing the frame, to average it. Done here instead,
//! the averaging is part of the frame the GPU is already working on and the
//! readback shrinks by `factor²` — 33 MB to 8 MB — because the copy now starts
//! from the small texture.
//!
//! **Averaging happens in linear light.** sRGB is a transfer function, not a
//! quantity you can take the mean of; averaging encoded bytes directly darkens
//! every edge, and a CAD render or a starfield is mostly edges. So the shader
//! decodes, averages, and re-encodes — matching what the CPU path did, and what
//! [`gpu_texture::MipGenerator`](super::gpu_texture::MipGenerator) gets for free
//! from the texture unit when it has a real sRGB format to work with.
//!
//! We cannot use that free path here: the renderer writes sRGB-*encoded* bytes
//! into a plain `Rgba8Unorm` target (its shader does the encode, see
//! `linear_fb` in `renderer.rs`), so the hardware has no idea the bytes are
//! encoded and would happily average them as-is. Six `pow` calls per output
//! pixel is nothing on a GPU; being wrong is not.

use wgpu::util::DeviceExt;

/// Averages a supersampled texture down to its output size in one render pass.
///
/// Owns its destination texture, sized `src / factor`. Build one per source
/// texture and keep it: everything it needs — pipeline, bind group, destination
/// — is created once, so [`resolve`](Self::resolve) allocates nothing.
///
/// ```no_run
/// # use threers::renderer::downsample::Downsampler;
/// # fn demo(device: &wgpu::Device, queue: &wgpu::Queue, hires: &wgpu::Texture) {
/// let ds = Downsampler::new(device, hires, 2).unwrap();
/// let mut enc = device.create_command_encoder(&Default::default());
/// ds.resolve(&mut enc);
/// queue.submit(Some(enc.finish()));
/// // ds.texture() now holds the averaged frame, ready to copy back.
/// # }
/// ```
pub struct Downsampler {
    factor: u32,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

/// Whether a format's bytes are sRGB-encoded and so must be decoded before
/// they can be averaged.
///
/// This mirrors `linear_fb` in `renderer.rs`: the mesh shader encodes sRGB for
/// every target except `Rgba16Float`, which stores linear light. Keeping the
/// same predicate — rather than the more obvious `format.is_srgb()` — is what
/// keeps the two from drifting apart.
fn holds_encoded_srgb(format: wgpu::TextureFormat) -> bool {
    format != wgpu::TextureFormat::Rgba16Float
}

impl Downsampler {
    /// Build a downsampler for `src`, averaging `factor × factor` blocks.
    ///
    /// `None` if `factor` is 1 (there is nothing to do), if the format cannot be
    /// rendered into, or if the source is not an exact multiple of `factor` —
    /// a partial block at the edge would be averaged against whatever the
    /// texture happens to hold there.
    pub fn new(device: &wgpu::Device, src: &wgpu::Texture, factor: u32) -> Option<Self> {
        let format = src.format();
        let (sw, sh) = (src.width(), src.height());
        if factor < 2 || !Self::renderable(format) || sw % factor != 0 || sh % factor != 0 {
            return None;
        }
        let (width, height) = (sw / factor, sh / factor);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers downsample"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers downsample layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // `textureLoad`, so no filtering is needed — and asking
                        // for none keeps float targets bindable.
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
        });

        // factor, srgb, and two words of padding: uniform buffers bind at 16.
        let params: [u32; 4] = [factor, holds_encoded_srgb(format) as u32, 0, 0];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers downsample params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Both transfer variants, so a later pass can view the same bytes as
        // linear or as sRGB. Bloom needs the sRGB view to get its decode for
        // free, and this texture is what it is handed whenever supersampling
        // is on — which left bloom panicking at view creation on any target
        // that was not already sRGB, i.e. the default one. `RenderTarget` has
        // always declared them; this did not.
        let mut extra_formats = Vec::new();
        for v in [format.add_srgb_suffix(), format.remove_srgb_suffix()] {
            if v != format && !extra_formats.contains(&v) {
                extra_formats.push(v);
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers downsample target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &extra_formats,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers downsample bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers downsample pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers downsample pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Some(Self {
            factor,
            width,
            height,
            texture,
            view,
            bind_group,
            pipeline,
        })
    }

    /// Formats that can be both loaded from and rendered into.
    fn renderable(format: wgpu::TextureFormat) -> bool {
        matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Rgba16Float
        )
    }

    /// Record the averaging pass. The result lands in [`texture`](Self::texture).
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers downsample pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Every pixel is written, so there is nothing to preserve.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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

    /// The averaged texture, at [`size`](Self::size).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
    /// Output size — the source size divided by [`factor`](Self::factor).
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// The block size being averaged.
    pub fn factor(&self) -> u32 {
        self.factor
    }
}

/// Average `factor × factor` blocks of RGBA8 down to one pixel, on the CPU.
///
/// The reference the GPU pass is tested against, and the fallback for the cases
/// [`Downsampler::new`] rejects. Prefer
/// [`HeadlessRenderer::render_to_rgba_resolved`](super::HeadlessRenderer::render_to_rgba_resolved),
/// which is about fifteen times faster; this is here so the contract holds even
/// where the GPU path cannot.
///
/// Colour is averaged in linear light, alpha as it stands — alpha carries no
/// transfer function.
pub fn average_blocks_srgb(src: &[u8], w: u32, h: u32, factor: u32) -> Vec<u8> {
    let (dw, dh) = (w / factor, h / factor);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let n = (factor * factor) as f32;
    let to_linear = |b: u8| {
        let s = b as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_srgb = |l: f32| {
        let s = if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = [0.0f32; 4];
            for sy in 0..factor {
                for sx in 0..factor {
                    let i = (((y * factor + sy) * w + (x * factor + sx)) * 4) as usize;
                    acc[0] += to_linear(src[i]);
                    acc[1] += to_linear(src[i + 1]);
                    acc[2] += to_linear(src[i + 2]);
                    acc[3] += src[i + 3] as f32 / 255.0;
                }
            }
            let o = ((y * dw + x) * 4) as usize;
            out[o] = to_srgb(acc[0] / n);
            out[o + 1] = to_srgb(acc[1] / n);
            out[o + 2] = to_srgb(acc[2] / n);
            out[o + 3] = ((acc[3] / n).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

const SHADER: &str = r#"
struct Params {
    factor : u32,
    srgb   : u32,
    _pad0  : u32,
    _pad1  : u32,
};

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var<uniform> p : Params;

// One oversized triangle rather than two: no seam down the diagonal.
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
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

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    // `pos` is the fragment centre, so truncating gives the output pixel.
    let base = vec2<i32>(pos.xy) * i32(p.factor);
    var acc = vec4<f32>(0.0);
    for (var y = 0u; y < p.factor; y = y + 1u) {
        for (var x = 0u; x < p.factor; x = x + 1u) {
            let at = base + vec2<i32>(i32(x), i32(y));
            var c = textureLoad(src, at, 0);
            // Alpha carries no transfer function: decode the colour only.
            if (p.srgb == 1u) { c = vec4<f32>(srgb_to_linear(c.rgb), c.a); }
            acc = acc + c;
        }
    }
    acc = acc / f32(p.factor * p.factor);
    if (p.srgb == 1u) { return vec4<f32>(linear_to_srgb(acc.rgb), acc.a); }
    return acc;
}
"#;
