//! Bloom for the headless path, as compute passes on the finished frame.
//!
//! Vacuum has no haze, so the only thing that spreads a highlight is the
//! camera: sunlight off machined aluminium, off a solar cell's coverglass, off
//! the Earth's limb. Every real orbital photograph has it and a render without
//! it reads as synthetic no matter how good the geometry is — the picture is
//! *too* clean, and the eye notices the absence before it can name it.
//!
//! # Why not `EffectComposer`
//!
//! threers already has a `BloomPass` and a composer to run it, and the composer
//! renders the scene into targets it owns. The headless renderer does not work
//! that way: it has its own multisampled target, a supersample resolve and a TAA
//! history, and routing the film through the composer would mean rebuilding all
//! of that on the other side. So this attaches where the frame is already
//! finished and already being read by a compute shader on its way to the
//! encoder, and leaves the render path alone.
//!
//! # Shape
//!
//! Threshold and downsample to a quarter on each axis, blur that separably,
//! and hand back a texture for the pack to add in. Blurring at a quarter
//! resolution is not a shortcut for its own sake — bloom is a low-frequency
//! effect, so the detail thrown away is detail the blur would remove anyway,
//! and it makes the kernel a sixteenth of the work.

/// Threshold + downsample + separable blur, producing a bloom texture.
pub struct Bloom {
    passes: Passes,
    /// The texture `view` points into. Held so the result can be copied out,
    /// which a view alone does not allow.
    tex_a: wgpu::Texture,
    view: wgpu::TextureView,
    src_size: (u32, u32),
}

/// The same three stages, recorded either way.
///
/// Compute is the shorter road and the one WebGPU takes. WebGL2 has no compute
/// stage at all, so there the stages are fullscreen draws instead — the
/// arithmetic is identical, and `bloom_paths_agree` holds the two to within a
/// rounding step of each other on the same device.
enum Passes {
    Compute {
        threshold_pipe: wgpu::ComputePipeline,
        blur_pipe: wgpu::ComputePipeline,
        bg_threshold: wgpu::BindGroup,
        bg_blur_h: wgpu::BindGroup,
        bg_blur_v: wgpu::BindGroup,
        groups: (u32, u32),
    },
    Raster {
        threshold_pipe: wgpu::RenderPipeline,
        blur_pipe: wgpu::RenderPipeline,
        bg_threshold: wgpu::BindGroup,
        bg_blur_h: wgpu::BindGroup,
        bg_blur_v: wgpu::BindGroup,
        /// Written by the threshold and the vertical blur; `view` aliases it.
        view_a: wgpu::TextureView,
        /// Written by the horizontal blur.
        view_b: wgpu::TextureView,
    },
}

/// One axis of the source is divided by this before blurring.
const SHRINK: u32 = 4;
const TILE: u32 = 8;

impl Bloom {
    /// `threshold` is the linear luminance above which a pixel blooms; `radius`
    /// scales the blur in low-resolution texels.
    pub fn new(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        threshold: f32,
        radius: f32,
    ) -> Option<Self> {
        // WebGL2 has no compute stage — wgpu signals that by zeroing the compute
        // limits rather than by refusing the device — so there the same three
        // stages are recorded as fullscreen draws instead.
        if crate::renderer::caps::supports_compute(device) {
            Self::new_compute(device, src, threshold, radius)
        } else {
            Self::new_raster(device, src, threshold, radius)
        }
    }

    /// The compute path, for devices that have the stage.
    pub(crate) fn new_compute(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        threshold: f32,
        radius: f32,
    ) -> Option<Self> {
        let (sw, sh) = (src.width(), src.height());
        let (bw, bh) = ((sw / SHRINK).max(1), (sh / SHRINK).max(1));
        // sRGB view: textureLoad then decodes to linear light for us, which is
        // the space a threshold and a blur mean anything in. Doing either on
        // stored sRGB values weights the highlights wrong — a mid grey is 0.5
        // encoded and 0.21 linear, and it is the linear number that says how
        // much light is actually there.
        let srgb = match src.format() {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                wgpu::TextureFormat::Rgba8UnormSrgb
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                wgpu::TextureFormat::Bgra8UnormSrgb
            }
            _ => return None,
        };
        let src_view = src.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers bloom src"),
            format: Some(srgb),
            ..Default::default()
        });

        let mk = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Half float, because bloom is about values ABOVE white and an
                // 8-bit target would clip the very highlights being gathered.
                format: wgpu::TextureFormat::Rgba16Float,
                // COPY_SRC so the result can be read back: the two paths are
                // compared against each other in `the_two_bloom_paths_agree`,
                // and a quarter-resolution texture costs nothing to allow it.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let tex_a = mk("threers bloom a");
        let tex_b = mk("threers bloom b");
        let view_a = tex_a.create_view(&Default::default());
        let view_b = tex_b.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers bloom"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // params: threshold, radius, direction x, direction y
        let ubo = |t: f32, r: f32, dx: f32, dy: f32| {
            let v = [t, r, dx, dy];
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers bloom params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            b.slice(..)
                .get_mapped_range_mut()
                .expect("buffer range is mapped")
                .slice(..)
                .copy_from_slice(bytemuck::cast_slice(&v));
            b.unmap();
            b
        };

        let sampled = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers bloom bgl"),
            entries: &[sampled(0), storage(1), uniform(2)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers bloom layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("threers bloom pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bind = |src: &wgpu::TextureView, dst: &wgpu::TextureView, u: wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers bloom bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: u.as_entire_binding(),
                    },
                ],
            })
        };

        Some(Self {
            passes: Passes::Compute {
                threshold_pipe: pipe("threshold"),
                blur_pipe: pipe("blur"),
                bg_threshold: bind(&src_view, &view_a, ubo(threshold, radius, 0.0, 0.0)),
                bg_blur_h: bind(&view_a, &view_b, ubo(threshold, radius, 1.0, 0.0)),
                bg_blur_v: bind(&view_b, &view_a, ubo(threshold, radius, 0.0, 1.0)),
                groups: (bw.div_ceil(TILE), bh.div_ceil(TILE)),
            },
            view: tex_a.create_view(&Default::default()),
            tex_a,
            src_size: (sw, sh),
        })
    }

    /// The same three stages as fullscreen draws, for devices with no compute.
    ///
    /// Two differences from the compute path, both forced:
    ///
    /// The bloom textures are render attachments rather than storage textures,
    /// since a fragment shader writes through the attachment and not through
    /// `textureStore`.
    ///
    /// And the source is decoded in the shader rather than by the sampler. The
    /// compute path views the frame as sRGB so that `textureLoad` returns linear
    /// light, which is the only space a luminance threshold means anything in —
    /// but reinterpreting a texture's format needs `VIEW_FORMATS`, which WebGL2
    /// does not have. So where the frame is stored in a linear format, the
    /// transfer function is applied in `fs_threshold` instead. Where it is
    /// already an sRGB format the sampler still does it, and the flag is off.
    pub(crate) fn new_raster(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        threshold: f32,
        radius: f32,
    ) -> Option<Self> {
        let (sw, sh) = (src.width(), src.height());
        let (bw, bh) = ((sw / SHRINK).max(1), (sh / SHRINK).max(1));
        let decode = match src.format() {
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb => 0.0,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm => 1.0,
            _ => return None,
        };
        let src_view = src.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers bloom src"),
            ..Default::default()
        });

        let mk = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Half float for the same reason as the compute path: bloom is
                // about values above white, and eight bits would clip exactly
                // the highlights being gathered.
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let tex_a = mk("threers bloom a");
        let tex_b = mk("threers bloom b");
        let view_a = tex_a.create_view(&Default::default());
        let view_b = tex_b.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers bloom (raster)"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{RASTER_SHADER}", crate::renderer::fullscreen::VS).into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers bloom raster bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Unfilterable: every tap is a `textureLoad`.
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers bloom raster layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers bloom raster pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // threshold, radius, direction x, direction y, decode, and padding to a
        // 32-byte block.
        let ubo = |t: f32, r: f32, dx: f32, dy: f32, dec: f32| {
            let v = [t, r, dx, dy, dec, 0.0, 0.0, 0.0f32];
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers bloom raster params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            b.slice(..)
                .get_mapped_range_mut()
                .expect("buffer range is mapped")
                .copy_from_slice(bytemuck::cast_slice(&v));
            b.unmap();
            b
        };
        let bind = |tex: &wgpu::TextureView, u: wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers bloom raster bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tex),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u.as_entire_binding(),
                    },
                ],
            })
        };

        Some(Self {
            passes: Passes::Raster {
                threshold_pipe: pipe("fs_threshold"),
                blur_pipe: pipe("fs_blur"),
                bg_threshold: bind(&src_view, ubo(threshold, radius, 0.0, 0.0, decode)),
                // The blur reads a half-float bloom texture, which is already
                // linear, so decoding is off for both of these.
                bg_blur_h: bind(&view_a, ubo(threshold, radius, 1.0, 0.0, 0.0)),
                bg_blur_v: bind(&view_b, ubo(threshold, radius, 0.0, 1.0, 0.0)),
                view_a,
                view_b,
            },
            view: tex_a.create_view(&Default::default()),
            tex_a,
            src_size: (sw, sh),
        })
    }

    /// Record threshold, horizontal blur, vertical blur. The result lands back
    /// in the first texture, so [`view`](Self::view) is stable.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        match &self.passes {
            Passes::Compute {
                threshold_pipe,
                blur_pipe,
                bg_threshold,
                bg_blur_h,
                bg_blur_v,
                groups,
            } => {
                for (pipe, bg, label) in [
                    (threshold_pipe, bg_threshold, "threers bloom threshold"),
                    (blur_pipe, bg_blur_h, "threers bloom blur h"),
                    (blur_pipe, bg_blur_v, "threers bloom blur v"),
                ] {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(label),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
            }
            Passes::Raster {
                threshold_pipe,
                blur_pipe,
                bg_threshold,
                bg_blur_h,
                bg_blur_v,
                view_a,
                view_b,
            } => {
                // The same ping-pong, written through attachments: threshold
                // into a, horizontal out of a into b, vertical out of b back
                // into a. The load op discards, because every pixel is written.
                for (pipe, bg, target, label) in [
                    (
                        threshold_pipe,
                        bg_threshold,
                        view_a,
                        "threers bloom threshold",
                    ),
                    (blur_pipe, bg_blur_h, view_b, "threers bloom blur h"),
                    (blur_pipe, bg_blur_v, view_a, "threers bloom blur v"),
                ] {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
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
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }
    }

    /// The blurred highlights, at 1/SHRINK of the source on each axis.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The texture [`view`](Self::view) points into.
    ///
    /// Carries `COPY_SRC`, so the result can be read back — which is how the two
    /// implementations are compared against each other.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.tex_a
    }

    /// Size of the frame this was built for, so a resize can be detected.
    pub fn src_size(&self) -> (u32, u32) {
        self.src_size
    }
}

const SHADER: &str = r#"
struct P { threshold : f32, radius : f32, dx : f32, dy : f32 };
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p : P;

// Threshold and downsample in one pass: average the SHRINK x SHRINK block that
// each output texel stands for, so a highlight one pixel wide still registers
// instead of being missed by point sampling.
@compute @workgroup_size(8, 8)
fn threshold(@builtin(global_invocation_id) gid : vec3<u32>) {
    let out_dim = textureDimensions(dst);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }
    let in_dim = textureDimensions(src);

    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let c = vec2<u32>(gid.x * 4u + i, gid.y * 4u + j);
            if (c.x >= in_dim.x || c.y >= in_dim.y) { continue; }
            // sRGB view: this is linear light already.
            let rgb = textureLoad(src, vec2<i32>(c), 0).rgb;
            let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            // Soft knee. A hard cut makes the bloom appear and vanish as a
            // highlight crosses the threshold, which on moving hardware reads
            // as flicker; scaling by how far past it the pixel is fades it in.
            let over = max(lum - p.threshold, 0.0);
            let w = over / max(lum, 1e-4);
            sum = sum + rgb * w;
            n = n + 1.0;
        }
    }
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(sum / max(n, 1.0), 1.0));
}

// Separable Gaussian, nine taps. Two passes of this is 18 samples for what a
// 2D kernel would spend 81 on, for the same result — the kernel is separable
// because a Gaussian is.
@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dim = textureDimensions(dst);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let dir = vec2<f32>(p.dx, p.dy) * p.radius;

    // sigma ~ 2.0 in texels
    var w = array<f32, 5>(0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162);
    var acc = textureLoad(src, vec2<i32>(gid.xy), 0).rgb * w[0];
    for (var k = 1; k < 5; k = k + 1) {
        let o = dir * f32(k);
        let a = clamp(vec2<i32>(gid.xy) + vec2<i32>(o), vec2<i32>(0), vec2<i32>(dim) - 1);
        let b = clamp(vec2<i32>(gid.xy) - vec2<i32>(o), vec2<i32>(0), vec2<i32>(dim) - 1);
        acc = acc + (textureLoad(src, a, 0).rgb + textureLoad(src, b, 0).rgb) * w[k];
    }
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(acc, 1.0));
}
"#;

/// The same arithmetic as [`SHADER`], as fullscreen draws.
///
/// Kept as a separate module rather than shared with the compute one because
/// the two differ in every place that touches the GPU — where the output goes,
/// how the source is decoded, how the invocation knows which pixel it is — and
/// a single source threaded with flags would be harder to check against the
/// other than two that can be read side by side.
const RASTER_SHADER: &str = r#"
struct P {
    threshold : f32,
    radius    : f32,
    dx        : f32,
    dy        : f32,
    // 1.0 when the source holds sRGB-encoded values in a linear format, so the
    // transfer function has to be applied here. The compute path gets this from
    // an sRGB texture view, which needs VIEW_FORMATS — and WebGL2 has none.
    decode    : f32,
    pad0      : f32,
    pad1      : f32,
    pad2      : f32,
};
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var<uniform> p : P;

// The exact piecewise sRGB curve, not a 2.2 power. The approximation is off by
// enough near black to move which pixels pass a luminance threshold.
fn to_linear(v : vec3<f32>) -> vec3<f32> {
    let lo = v / 12.92;
    let hi = pow((v + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, v <= vec3<f32>(0.04045));
}

fn fetch_src(c : vec2<i32>) -> vec3<f32> {
    let v = textureLoad(src, c, 0).rgb;
    if (p.decode > 0.5) {
        return to_linear(v);
    }
    return v;
}

// Threshold and downsample in one pass, averaging the 4x4 block each output
// texel stands for — so a highlight one pixel wide still registers instead of
// being missed by point sampling.
@fragment
fn fs_threshold(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    // `position` is the pixel centre, so truncating gives the texel index.
    let gid = vec2<u32>(pos.xy);
    let in_dim = textureDimensions(src);
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let c = vec2<u32>(gid.x * 4u + i, gid.y * 4u + j);
            if (c.x >= in_dim.x || c.y >= in_dim.y) { continue; }
            let rgb = fetch_src(vec2<i32>(c));
            let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            // Soft knee, as in the compute path: a hard cut makes bloom appear
            // and vanish as a highlight crosses the threshold, which reads as
            // flicker.
            let over = max(lum - p.threshold, 0.0);
            let w = over / max(lum, 1e-4);
            sum = sum + rgb * w;
            n = n + 1.0;
        }
    }
    return vec4<f32>(sum / max(n, 1.0), 1.0);
}

// Separable Gaussian, nine taps, same weights as the compute path. The source
// here is always a half-float bloom texture, so it is linear already and
// `decode` is off.
@fragment
fn fs_blur(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let gid = vec2<i32>(vec2<u32>(pos.xy));
    let dim = vec2<i32>(textureDimensions(src));
    let dir = vec2<f32>(p.dx, p.dy) * p.radius;
    var w = array<f32, 5>(0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162);
    var acc = textureLoad(src, gid, 0).rgb * w[0];
    for (var k = 1; k < 5; k = k + 1) {
        let o = dir * f32(k);
        let a = clamp(gid + vec2<i32>(o), vec2<i32>(0), dim - vec2<i32>(1));
        let b = clamp(gid - vec2<i32>(o), vec2<i32>(0), dim - vec2<i32>(1));
        acc = acc + (textureLoad(src, a, 0).rgb + textureLoad(src, b, 0).rgb) * w[k];
    }
    return vec4<f32>(acc, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The compute and raster paths produce the same bloom.
    ///
    /// This is the check that matters for the raster path: it exists so WebGL2
    /// can bloom, and WebGL2 is the one place it cannot be compared against the
    /// implementation it is meant to reproduce. So the two are compared *here*,
    /// on a device that has both, where any difference is the port and not the
    /// backend.
    ///
    /// Agreement has to be close rather than exact. Both write `Rgba16Float`,
    /// but one arrives through `textureStore` and the other through a colour
    /// attachment, and the fragment path's `@builtin(position)` is a float that
    /// gets truncated to the texel index — so a tolerance of a few thousandths
    /// is the honest bar. A wrong threshold, a wrong weight or a dropped tap
    /// moves things far more than that.
    #[test]
    fn the_two_bloom_paths_agree() {
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        // A frame with a bright patch on a dim field: the patch blooms, the
        // field does not, so the result exercises both sides of the threshold.
        const W: u32 = 64;
        let mut bytes = vec![0u8; (W * W * 4) as usize];
        for y in 0..W {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                let bright = (20..32).contains(&x) && (20..32).contains(&y);
                let v = if bright { 250 } else { 40 };
                bytes[i] = v;
                bytes[i + 1] = v / 2;
                bytes[i + 2] = v / 4;
                bytes[i + 3] = 255;
            }
        }
        let src = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom test src"),
            size: wgpu::Extent3d {
                width: W,
                height: W,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            // Both the sRGB view the compute path takes and the plain one the
            // raster path takes have to be available from this one texture.
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(W),
            },
            wgpu::Extent3d {
                width: W,
                height: W,
                depth_or_array_layers: 1,
            },
        );

        let compute = Bloom::new_compute(&device, &src, 0.2, 1.0).expect("compute bloom");
        let raster = Bloom::new_raster(&device, &src, 0.2, 1.0).expect("raster bloom");
        let a = run_and_read(&device, &queue, &compute, W / SHRINK);
        let b = run_and_read(&device, &queue, &raster, W / SHRINK);
        assert_eq!(a.len(), b.len());

        let mut worst = 0.0f32;
        let mut sum_a = 0.0f32;
        let mut peak = 0.0f32;
        for (x, y) in a.iter().zip(&b) {
            worst = worst.max((x - y).abs());
            sum_a += x.abs();
            peak = peak.max(x.abs());
        }
        // The bloom must not be empty, or the comparison is vacuous.
        assert!(
            sum_a > 1.0,
            "the compute bloom is blank ({sum_a}), so agreement proves nothing"
        );
        // The two agree to the last representable bit: measured worst case is
        // 0.000244, which is one half-float step at this magnitude. The bar is a
        // few steps rather than a few hundredths — a loose tolerance here passed
        // a deliberately wrong blur weight, which is exactly the mistake this
        // test exists to catch.
        assert!(
            worst < 0.001 * peak.max(1.0),
            "the two bloom paths differ by {worst} at worst (peak {peak}) — the \
             raster port does not reproduce the compute one"
        );
    }

    /// Run a bloom and read its texture back as floats.
    fn run_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bloom: &Bloom,
        side: u32,
    ) -> Vec<f32> {
        let bytes_per_row = (side * 8).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom readback"),
            size: (bytes_per_row * side) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        bloom.record(&mut encoder);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: bloom.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(side),
                },
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = readback
            .slice(..)
            .get_mapped_range()
            .expect("buffer range is mapped");
        let mut out = Vec::with_capacity((side * side * 4) as usize);
        for row in 0..side {
            let start = (row * bytes_per_row) as usize;
            let end = start + (side * 8) as usize;
            for pair in data[start..end].chunks_exact(2) {
                out.push(half_to_f32(u16::from_le_bytes([pair[0], pair[1]])));
            }
        }
        out
    }

    /// IEEE half to float, enough for reading back a render target.
    fn half_to_f32(h: u16) -> f32 {
        crate::renderer::readback::f16_to_f32(h)
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
            label: Some("bloom test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }
}
