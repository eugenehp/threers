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
    threshold_pipe: wgpu::ComputePipeline,
    blur_pipe: wgpu::ComputePipeline,
    bg_threshold: wgpu::BindGroup,
    bg_blur_h: wgpu::BindGroup,
    bg_blur_v: wgpu::BindGroup,
    view: wgpu::TextureView,
    groups: (u32, u32),
    src_size: (u32, u32),
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
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
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
                .get_mapped_range_mut().expect("buffer range is mapped")
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
            threshold_pipe: pipe("threshold"),
            blur_pipe: pipe("blur"),
            bg_threshold: bind(&src_view, &view_a, ubo(threshold, radius, 0.0, 0.0)),
            bg_blur_h: bind(&view_a, &view_b, ubo(threshold, radius, 1.0, 0.0)),
            bg_blur_v: bind(&view_b, &view_a, ubo(threshold, radius, 0.0, 1.0)),
            view: tex_a.create_view(&Default::default()),
            groups: (bw.div_ceil(TILE), bh.div_ceil(TILE)),
            src_size: (sw, sh),
        })
    }

    /// Record threshold, horizontal blur, vertical blur. The result lands back
    /// in the first texture, so [`view`](Self::view) is stable.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        for (pipe, bg, label) in [
            (
                &self.threshold_pipe,
                &self.bg_threshold,
                "threers bloom threshold",
            ),
            (&self.blur_pipe, &self.bg_blur_h, "threers bloom blur h"),
            (&self.blur_pipe, &self.bg_blur_v, "threers bloom blur v"),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
        }
    }

    /// The blurred highlights, at 1/SHRINK of the source on each axis.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
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
