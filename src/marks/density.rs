//! Density fields: kernels summed offscreen, then mapped through a ramp.

use super::{Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, PrepareContext};
use crate::math::ColorRamp;

/// The field's format. Half float sums thousands of overlapping kernels
/// without saturating and is a blendable render target on WebGL2 as well
/// (`EXT_color_buffer_float`), which a 32-bit float is not without a further
/// extension.
pub(crate) const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// A heatmap: every row adds a Gaussian kernel to a field, and the field is
/// coloured through a ramp — deck.gl's `HeatmapLayer`.
///
/// Two passes. The kernels are summed additively into an offscreen float
/// target; then one fullscreen pass samples that field, filtered, maps each
/// pixel's sum through [`ramp`](Self::set_ramp) and blends it over the frame.
/// Both are render passes, so it runs on WebGL2.
///
/// The sum is smooth, so the field need not be as sharp as the screen:
/// by default it is just fine enough for a kernel to span twelve texels of
/// radius, where it stays within 2 % of the full-resolution field, and never
/// coarser than a quarter of the viewport. Kernels 30 px in radius then sum
/// into a field with a sixth of the screen's pixels, 48 px and over a
/// sixteenth. [`resolution`](Self::resolution) overrides the choice.
///
/// The radius is [`MarkStyle::size_scale`](super::MarkStyle::size_scale), in
/// pixels or world units; each row's weight is its `weights` entry (1 by
/// default). The sum that reaches the top of the ramp is
/// [`max_weight`](Self::max_weight) — fixed rather than measured, so the colour
/// of a place does not change as other places scroll in and out of view — or,
/// with [`auto_max`](Self::auto_max), the densest pixel on screen, measured on
/// the GPU each frame and never read back.
#[derive(Debug)]
pub struct DensityMarks {
    pub common: MarkCommon,
    positions: Column,
    weights: Column,
    /// The summed weight that maps to the top of the ramp.
    pub max_weight: f32,
    /// Scale to the field instead of [`max_weight`](Self::max_weight): the
    /// top of the ramp is this fraction of the densest pixel drawn this
    /// frame — deck.gl's automatic colour domain. 1 puts the peak at the top;
    /// less saturates the peaks to show the sparse parts. The colour of a
    /// place then depends on what else is in view.
    pub auto_max: Option<f32>,
    /// Fraction of the ramp over which the field fades in from transparent.
    pub threshold: f32,
    /// The field's size as a fraction of the viewport's, per axis, in
    /// `0.05..=1`. `None` picks it from the kernel size: full resolution for
    /// kernels up to 12 px in radius, down to a quarter from 48 px.
    pub resolution: Option<f32>,
    ramp: Vec<[u8; 4]>,
    ramp_dirty: bool,
    gpu: Option<DensityGpu>,
}

#[derive(Debug)]
struct DensityGpu {
    size: (u32, u32),
    field: wgpu::TextureView,
    ramp: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// The max reduction, field to one texel.
    levels: Vec<ReduceLevel>,
}

/// One step of the field's max reduction.
#[doc(hidden)]
#[derive(Debug)]
pub struct ReduceLevel {
    pub(crate) target: wgpu::TextureView,
    /// Reads the level before (the field, for the first).
    pub(crate) bind_group: wgpu::BindGroup,
}

/// Each level a quarter of the last per axis, down to one texel.
fn reduce_sizes(mut size: (u32, u32)) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    loop {
        size = (size.0.div_ceil(4).max(1), size.1.div_ceil(4).max(1));
        out.push(size);
        if size == (1, 1) {
            return out;
        }
    }
}

impl Default for DensityMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl DensityMarks {
    pub fn new() -> Self {
        let mut weights = Column::float(1);
        weights.set_default_f32(&[1.0]);
        let mut common = MarkCommon::default();
        common.style.size_scale = 30.0;
        common.style.depth_test = false;
        common.style.pickable = false;
        let mut s = Self {
            common,
            positions: Column::new(ColumnFormat::PositionF64),
            weights,
            max_weight: 4.0,
            auto_max: None,
            threshold: 0.05,
            resolution: None,
            ramp: Vec::new(),
            ramp_dirty: true,
            gpu: None,
        };
        s.set_ramp(&ColorRamp::inferno());
        s
    }

    /// The field's size relative to the viewport this frame.
    pub fn field_scale(&self) -> f32 {
        if let Some(r) = self.resolution {
            return r.clamp(0.05, 1.0);
        }
        let s = &self.common.style;
        if s.units == super::SizeUnits::World {
            // The kernel's size on screen varies with depth; split the
            // difference.
            return 0.5;
        }
        let radius = s.size_scale.clamp(s.min_pixels, s.max_pixels);
        // Twelve texels across the radius: filtering between coarse texels
        // shaves a lone kernel's peak by about 4.5 / (2 n²), 1.6 % here.
        (12.0 / radius.max(1e-3)).clamp(0.25, 1.0)
    }

    pub fn set_positions(&mut self, p: &[[f64; 3]]) {
        self.common.local_origin = None;
        self.positions.set_positions(p);
    }
    /// Positions relative to `origin` — half the memory of absolute ones,
    /// and what a tile decoder produces. See
    /// [`ColumnFormat::PositionLocal`](super::ColumnFormat::PositionLocal).
    pub fn set_local_positions(&mut self, origin: [f64; 3], p: &[[f32; 3]]) {
        self.common.local_origin = Some(origin);
        self.positions.set_local_positions(p);
    }
    pub fn update_local_positions(&mut self, start: usize, p: &[[f32; 3]]) {
        self.positions.update_local_positions(start, p);
    }
    pub fn update_positions(&mut self, start: usize, p: &[[f64; 3]]) {
        self.positions.update_positions(start, p);
    }
    pub fn set_weights(&mut self, w: &[f32]) {
        self.weights.set_f32(w);
    }
    pub fn update_weights(&mut self, start: usize, w: &[f32]) {
        self.weights.update_f32(start, w);
    }

    /// The colours from no density to [`max_weight`](Self::max_weight).
    pub fn set_ramp(&mut self, ramp: &ColorRamp) {
        self.ramp = ramp
            .sample(256)
            .into_iter()
            .map(|c| {
                let e = |v: f32| {
                    let v = v.clamp(0.0, 1.0);
                    let s = if v <= 0.003_130_8 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    };
                    (s * 255.0 + 0.5) as u8
                };
                [e(c.r), e(c.g), e(c.b), 255]
            })
            .collect();
        self.ramp_dirty = true;
    }
}

impl Mark for DensityMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Splat
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        self.positions.len()
    }
    fn kind_uniforms(&self) -> KindUniforms {
        // style3: what the composite maps to the ramp.
        let (auto, fraction) = match self.auto_max {
            Some(k) => (1.0, k.max(1e-6)),
            None => (0.0, 1.0),
        };
        (
            [0.0; 4],
            [self.max_weight, self.threshold, auto, fraction],
            [0.0; 4],
        )
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        let scale = self.field_scale();
        let vp = ctx.view.viewport;
        let size = (
            ((vp.0 as f32 * scale).ceil() as u32).max(1),
            ((vp.1 as f32 * scale).ceil() as u32).max(1),
        );
        let stale = self.gpu.as_ref().is_none_or(|g| g.size != size);
        if stale {
            let field = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers density field"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FIELD_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let ramp = match self.gpu.take() {
                Some(g) => g.ramp,
                None => ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("threers density ramp"),
                    size: wgpu::Extent3d {
                        width: 256,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // sRGB bytes; sampling decodes them to linear.
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                }),
            };
            let field_view = field.create_view(&Default::default());
            let ramp_view = ramp.create_view(&Default::default());
            let mut levels = Vec::new();
            let mut src = field_view.clone();
            for (w, h) in reduce_sizes(size) {
                let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("threers density max"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FIELD_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let target = tex.create_view(&Default::default());
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers density reduce bind group"),
                    layout: &ctx.resources.reduce_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &ctx.resources.nearest_sampler,
                            ),
                        },
                    ],
                });
                src = target.clone();
                levels.push(ReduceLevel { target, bind_group });
            }
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers density composite bind group"),
                layout: &ctx.resources.density_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&field_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&ramp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&ctx.resources.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&src),
                    },
                ],
            });
            self.gpu = Some(DensityGpu {
                size,
                field: field_view,
                ramp,
                bind_group,
                levels,
            });
        }
        if self.ramp_dirty || stale {
            let g = self.gpu.as_ref().unwrap();
            let bytes: Vec<u8> = self.ramp.iter().flatten().copied().collect();
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &g.ramp,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.ramp_dirty = false;
        }
    }
    fn release_gpu(&mut self) {
        self.gpu = None;
        self.ramp_dirty = true;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        // The field: four half floats a texel.
        let field = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.size.0 as u64 * g.size.1 as u64 * 8);
        super::column_bytes(self) + field
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![&mut self.positions, &mut self.weights]
    }
    fn density(&self) -> Option<(&wgpu::TextureView, &wgpu::BindGroup)> {
        self.gpu.as_ref().map(|g| (&g.field, &g.bind_group))
    }
    fn density_reduction(&self) -> Option<&[ReduceLevel]> {
        self.auto_max?;
        self.gpu.as_ref().map(|g| g.levels.as_slice())
    }
}
