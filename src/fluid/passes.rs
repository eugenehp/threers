//! The GPU side: the textures the fluid lives in, the uniform block every pass
//! reads, and the pipelines that write them.
//!
//! Nothing here is specific to a fluid — it is the machinery for running a
//! sequence of fullscreen fragment passes over ping-ponged float targets.
//! [`super::FluidSim`] is what sequences them.

use crate::renderer::fullscreen::FormatPipelines;

/// The uniform block, mirroring `Params` in [`super::shader`].
///
/// Field order is the layout: WGSL puts a `vec4` on a 16-byte boundary, so the
/// four loose scalars sit between the two `vec2` pairs and the two `vec4`s
/// rather than after them. `params_layout_matches_wgsl` pins the result.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Params {
    pub texel_size: [f32; 2],
    pub point: [f32; 2],
    pub dye_texel_size: [f32; 2],
    pub dt: f32,
    pub dissipation: f32,
    pub curl: f32,
    pub aspect_ratio: f32,
    /// WGSL puts the next `vec4` on a 16-byte boundary and `repr(C)` would
    /// not; without this the two layouts diverge at `color` and every pass
    /// reads the wrong eight bytes.
    pub pad0: [f32; 2],
    pub color: [f32; 4],
    pub curve: [f32; 4],
    pub radius: f32,
    pub value: f32,
    pub flags: u32,
    pub pad: u32,
}

pub(crate) const FLAG_SHADING: u32 = 1;
pub(crate) const FLAG_BLOOM: u32 = 2;
pub(crate) const FLAG_SUNRAYS: u32 = 4;
pub(crate) const FLAG_ENCODE: u32 = 8;

/// Bytes per uniform block. WGSL rounds the struct up to its own alignment.
pub(crate) const PARAMS_SIZE: u64 = 96;

/// Splats in one pass. Two `vec4`s each, so the block is 1 KiB — well inside
/// the 64 KiB a uniform buffer is guaranteed, and more than a hand has
/// fingers or a burst has blobs.
pub(crate) const MAX_BATCH: usize = 32;

/// Bytes in the splat batch: `array<vec4<f32>, 64>`.
pub(crate) const BATCH_SIZE: u64 = (MAX_BATCH as u64) * 2 * 16;

/// Which float formats the fields are stored in.
///
/// The simulation needs one, two and four channel float targets that can be
/// both rendered to and filtered. WebGPU guarantees all three; WebGL2 through
/// wgpu's GL backend guarantees none of them without `EXT_color_buffer_float`,
/// and a device that has only the four-channel one still runs the whole
/// simulation — it just carries three unused channels in the pressure field.
/// So the narrow formats are a saving, not a requirement, and
/// [`FluidFormats::for_adapter`] gives up each one independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidFormats {
    /// Dye — three colour channels and nothing in alpha.
    pub dye: wgpu::TextureFormat,
    /// The *volume's* dye, which can be packed tighter than the plane's.
    ///
    /// Three channels of 11-bit float in one word: half the memory and half
    /// the bandwidth of four channels of 16, and the dye grid is both the
    /// largest field in a volume and the one the heaviest pass reads and
    /// writes. It is unsigned, which the plane's dye could not be — a
    /// MacCormack correction is an extrapolation and can undershoot — but
    /// nothing in the volume subtracts from the dye.
    ///
    /// `rg11b10ufloat` is only a render target where the device has that
    /// feature, so this falls back to [`dye`](Self::dye).
    pub volume_dye: wgpu::TextureFormat,
    /// Velocity — two channels.
    pub velocity: wgpu::TextureFormat,
    /// Pressure, divergence, curl, sunrays — one channel.
    pub scalar: wgpu::TextureFormat,
}

impl Default for FluidFormats {
    /// What WebGPU's baseline guarantees.
    fn default() -> Self {
        Self {
            dye: wgpu::TextureFormat::Rgba16Float,
            // Not the packed one: it is a render target only where the
            // device asked for the feature, and this is the baseline.
            // `FluidSim3d` upgrades it when the device it is handed can.
            volume_dye: wgpu::TextureFormat::Rgba16Float,
            velocity: wgpu::TextureFormat::Rg16Float,
            scalar: wgpu::TextureFormat::R16Float,
        }
    }
}

impl FluidFormats {
    /// The formats a volume will *actually* allocate, given its
    /// `compact_dye`.
    ///
    /// [`volume_dye`](Self::volume_dye) is what the device would allow, not
    /// what the simulation asked for, and those are different things: the
    /// packed format is off by default. Sizing a volume against the first
    /// while it allocates with the second underestimates the dye — which is
    /// most of the memory — by a factor of two. Resolve before predicting.
    pub fn resolve(self, compact_dye: bool) -> Self {
        Self {
            volume_dye: if compact_dye {
                self.volume_dye
            } else {
                self.dye
            },
            ..self
        }
    }

    /// The narrowest formats this adapter will render to and filter, widening
    /// one channel count at a time: `r16float` → `rg16float` → `rgba16float`.
    /// Returns `None` if even `rgba16float` is unusable, which means the
    /// simulation cannot run on this device at all.
    pub fn for_adapter(adapter: &wgpu::Adapter) -> Option<Self> {
        use wgpu::TextureFormat as F;
        let usable = |f: F| {
            let feats = adapter.get_texture_format_features(f);
            feats
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
                && feats
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        };
        if !usable(F::Rgba16Float) {
            return None;
        }
        Some(Self {
            dye: F::Rgba16Float,
            volume_dye: if usable(F::Rg11b10Ufloat) {
                F::Rg11b10Ufloat
            } else {
                F::Rgba16Float
            },
            velocity: if usable(F::Rg16Float) {
                F::Rg16Float
            } else {
                F::Rgba16Float
            },
            scalar: if usable(F::R16Float) {
                F::R16Float
            } else if usable(F::Rg16Float) {
                F::Rg16Float
            } else {
                F::Rgba16Float
            },
        })
    }
}

/// Blend `src + dst * (1 - src.a)` — the dye compositing over the background.
const OVER: wgpu::BlendState = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

/// Blend `src + dst`, for the bloom chain's upsample.
const ADD: wgpu::BlendState = wgpu::BlendState {
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
};

/// Everything compiled once: the layouts, the module, and one pipeline per
/// (stage, target format, blend) the fixed part of the frame needs.
pub(crate) struct Pipelines {
    pub bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,

    pub copy_dye: wgpu::RenderPipeline,
    pub copy_velocity: wgpu::RenderPipeline,
    pub clear: wgpu::RenderPipeline,
    pub fill_velocity: wgpu::RenderPipeline,
    pub splat_velocity: wgpu::RenderPipeline,
    pub splat_dye: wgpu::RenderPipeline,
    pub advect_velocity: wgpu::RenderPipeline,
    pub advect_dye: wgpu::RenderPipeline,
    pub correct_velocity: wgpu::RenderPipeline,
    pub correct_dye: wgpu::RenderPipeline,
    pub divergence: wgpu::RenderPipeline,
    pub curl: wgpu::RenderPipeline,
    pub vorticity: wgpu::RenderPipeline,
    pub pressure: wgpu::RenderPipeline,
    pub gradient: wgpu::RenderPipeline,
    pub residual: wgpu::RenderPipeline,
    pub restrict: wgpu::RenderPipeline,
    pub prolong: wgpu::RenderPipeline,
    pub bloom_prefilter: wgpu::RenderPipeline,
    pub bloom_blur: wgpu::RenderPipeline,
    pub bloom_blur_add: wgpu::RenderPipeline,
    pub bloom_final: wgpu::RenderPipeline,
    pub sunrays_mask: wgpu::RenderPipeline,
    pub sunrays: wgpu::RenderPipeline,
    pub blur: wgpu::RenderPipeline,

    /// Display, background and checkerboard, keyed by the format of whatever
    /// the caller is rendering into. A surface, an offscreen target and a
    /// readback buffer are three different formats, and the fluid should not
    /// care which it was handed.
    out: FormatPipelines<OutputPipes>,
}

/// The three pipelines whose target format is the caller's.
pub(crate) struct OutputPipes {
    pub display_over: wgpu::RenderPipeline,
    pub display_replace: wgpu::RenderPipeline,
    pub color: wgpu::RenderPipeline,
    pub checkerboard: wgpu::RenderPipeline,
    /// The frame-rate read-out, composited over the finished picture.
    pub overlay: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device, formats: FluidFormats) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers fluid bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The splat batch. Bound for every pass, because one layout
                // serves them all; the others never read it.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(BATCH_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers fluid layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers fluid"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", super::read_out::glyphs(), super::shader::SHADER).into(),
            ),
        });

        // The plane does not use `volume_dye`; see its docs for why the
        // volume can pack its dye and this cannot.
        let FluidFormats {
            dye,
            velocity,
            scalar,
            volume_dye: _,
        } = formats;
        let p =
            |fs: &str, format, blend| build(device, &layout, &module, "vs", fs, format, blend, fs);
        Self {
            copy_dye: p("fs_copy", dye, None),
            copy_velocity: p("fs_copy", velocity, None),
            clear: p("fs_clear", scalar, None),
            fill_velocity: p("fs_color", velocity, None),
            splat_velocity: p("fs_splat", velocity, None),
            splat_dye: p("fs_splat", dye, None),
            advect_velocity: p("fs_advection", velocity, None),
            advect_dye: p("fs_advection", dye, None),
            correct_velocity: p("fs_advect_correct", velocity, None),
            correct_dye: p("fs_advect_correct", dye, None),
            divergence: p("fs_divergence", scalar, None),
            curl: p("fs_curl", scalar, None),
            vorticity: p("fs_vorticity", velocity, None),
            pressure: p("fs_pressure", scalar, None),
            gradient: p("fs_gradient", velocity, None),
            residual: p("fs_residual", scalar, None),
            restrict: p("fs_restrict", scalar, None),
            prolong: p("fs_prolong", scalar, None),
            bloom_prefilter: p("fs_bloom_prefilter", dye, None),
            bloom_blur: p("fs_bloom_blur", dye, None),
            bloom_blur_add: p("fs_bloom_blur", dye, Some(ADD)),
            bloom_final: p("fs_bloom_final", dye, None),
            // The mask is written into the dye's spare buffer, so it is in the
            // dye's format; the rays themselves are one channel.
            sunrays_mask: p("fs_sunrays_mask", dye, None),
            sunrays: p("fs_sunrays", scalar, None),
            blur: build(
                device, &layout, &module, "vs_blur", "fs_blur", scalar, None, "fs_blur",
            ),
            bgl,
            layout,
            module,
            out: FormatPipelines::new(),
        }
    }

    /// The output pipelines for a target format, compiled on first sight.
    pub fn output(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &OutputPipes {
        self.out.get_or_build(format, || {
            let p = |fs: &str, blend| {
                build(
                    device,
                    &self.layout,
                    &self.module,
                    "vs",
                    fs,
                    format,
                    blend,
                    fs,
                )
            };
            OutputPipes {
                display_over: p("fs_display", Some(OVER)),
                display_replace: p("fs_display", None),
                color: p("fs_color", None),
                checkerboard: p("fs_checkerboard", None),
                overlay: p("fs_overlay", Some(wgpu::BlendState::ALPHA_BLENDING)),
            }
        })
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
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
}

/// One uniform block, in a buffer of its own.
///
/// A pass's parameters have to be visible to the GPU when that pass runs, and
/// `Queue::write_buffer` only orders writes against *submits* — every write
/// issued before a submit lands before all of its passes. Two passes in one
/// submit therefore cannot share a buffer, so each takes its own. They are
/// eighty bytes; there are about thirty of them.
pub(crate) fn ubo(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    sized_ubo(device, label, PARAMS_SIZE)
}

/// The splat batch's buffer, or the zeroed stand-in every other pass binds.
pub(crate) fn batch_ubo(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    sized_ubo(device, label, BATCH_SIZE)
}

fn sized_ubo(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A bind group over one uniform block and up to three textures. Passes that
/// read fewer get the same view in the unused slots — see [`super::shader`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    params: &wgpu::Buffer,
    batch: &wgpu::Buffer,
    a: &wgpu::TextureView,
    b: Option<&wgpu::TextureView>,
    c: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("threers fluid bind"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(a),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(b.unwrap_or(a)),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(c.unwrap_or(a)),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: batch.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust block and the WGSL one have to agree byte for byte, and nothing
    /// in either language will say so if they drift — the GPU reads whatever is
    /// at the offset and the picture goes subtly wrong. The offsets below are
    /// WGSL's, derived from its alignment rules; this holds Rust to them.
    #[test]
    fn params_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<Params>() as u64, PARAMS_SIZE);
        let p = Params::default();
        let base = &p as *const Params as usize;
        let at = |f: *const ()| f as usize - base;
        assert_eq!(at(&p.texel_size as *const _ as *const ()), 0);
        assert_eq!(at(&p.point as *const _ as *const ()), 8);
        assert_eq!(at(&p.dye_texel_size as *const _ as *const ()), 16);
        assert_eq!(at(&p.dt as *const _ as *const ()), 24);
        assert_eq!(at(&p.dissipation as *const _ as *const ()), 28);
        assert_eq!(at(&p.curl as *const _ as *const ()), 32);
        assert_eq!(at(&p.aspect_ratio as *const _ as *const ()), 36);
        // The two vec4s, which are why the scalars sit where they do.
        assert_eq!(at(&p.color as *const _ as *const ()), 48);
        assert_eq!(at(&p.curve as *const _ as *const ()), 64);
        assert_eq!(at(&p.radius as *const _ as *const ()), 80);
        assert_eq!(at(&p.value as *const _ as *const ()), 84);
        assert_eq!(at(&p.flags as *const _ as *const ()), 88);
    }
}
