//! Cross-sections: an arbitrary plane through a [`GpuVolume`], drawn into a
//! rectangle of the target.
//!
//! Each draw is a fullscreen triangle confined to the rectangle. The fragment
//! stage maps its NDC position onto the plane and into voxel coordinates with
//! one matrix ([`plane_to_voxel`] builds the usual one), then reads the
//! nearest texel with `textureLoad` and colours it. Outside the volume it
//! writes nothing, so whatever is underneath shows through.
//!
//! Nearest, because a slice viewer is usually asked what is *in* a voxel, and
//! because `textureLoad` needs no filtering support, which `R32Float` and the
//! integer formats do not have everywhere.
//!
//! # Colour
//!
//! A WGSL function turns what was read into a colour, with four `vec4` of
//! parameters that can change per draw:
//!
//! ```wgsl
//! // VolumeSample::Float
//! fn slice_color(value : vec4<f32>, params : array<vec4<f32>, 4>) -> vec4<f32>
//! // VolumeSample::Uint
//! fn slice_color(value : vec4<u32>, params : array<vec4<f32>, 4>) -> vec4<f32>
//! ```
//!
//! [`WINDOW_LEVEL`] (grey ramp, `params[0].xy` = black and white points) and
//! [`LABELS`] (a stable colour per label, 0 transparent) cover the common
//! cases. Return display-referred colour; the pass linearises it when the
//! target is sRGB, so a 0.5 grey shows as 0.5 on either kind of target.
//!
//! ```no_run
//! use threers::volume::{plane_to_voxel, Data3DTexture, GpuVolume, SliceParams, SliceRenderer};
//! use threers::math::{Quaternion, Vector3};
//! use threers::renderer::PixelRect;
//! # fn demo(device: &wgpu::Device, queue: &wgpu::Queue, frame: &wgpu::TextureView) {
//! let data = Data3DTexture::from_f32(64, 64, 64, &vec![0.5; 64 * 64 * 64]);
//! let volume = GpuVolume::from_texture(device, queue, &data);
//! let slices = SliceRenderer::window_level(device, wgpu::TextureFormat::Bgra8UnormSrgb);
//! let panel = slices.create_binding(device, &volume);
//!
//! let rect = PixelRect { x: 0, y: 0, width: 400, height: 300 };
//! let to_voxel = plane_to_voxel(Vector3::new(32.0, 32.0, 32.0), Quaternion::identity(), 0.25, (400.0, 300.0), true);
//! slices.update(queue, &panel, &SliceParams::window(to_voxel, 0.0, 1.0));
//! let mut encoder = device.create_command_encoder(&Default::default());
//! slices.draw(&mut encoder, frame, rect, &panel);
//! queue.submit([encoder.finish()]);
//! # }
//! ```

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use super::texture::{GpuVolume, VolumeSample};
use crate::math::{Matrix4, Quaternion, Vector3};
use crate::renderer::fullscreen;
use crate::renderer::PixelRect;

/// Grey ramp over `value.r`: `params[0].x` maps to black, `params[0].y` to
/// white. For [`VolumeSample::Float`] volumes.
pub const WINDOW_LEVEL: &str = r#"
fn slice_color(value : vec4<f32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    let g = clamp((value.r - params[0].x) / (params[0].y - params[0].x), 0.0, 1.0);
    return vec4<f32>(g, g, g, 1.0);
}
"#;

/// A colour per label, from a hash of the 64-bit value in `.r` (low word) and
/// `.g` (high word); label 0 is transparent. `params[0].x` is a seed that
/// reshuffles the colours, `params[0].y` the opacity. For
/// [`VolumeSample::Uint`] volumes.
pub const LABELS: &str = r#"
fn slice_label_hash(seed : u32, v : u32) -> u32 {
    // MurmurHash3's block mix.
    var k = v * 0xcc9e2d51u;
    k = (k << 15u) | (k >> 17u);
    k = k * 0x1b873593u;
    var h = seed ^ k;
    h = (h << 13u) | (h >> 19u);
    return h * 5u + 0xe6546b64u;
}
fn slice_color(value : vec4<u32>, params : array<vec4<f32>, 4>) -> vec4<f32> {
    if (value.r == 0u && value.g == 0u) { return vec4<f32>(0.0); }
    let h = slice_label_hash(slice_label_hash(u32(params[0].x), value.r), value.g);
    let hue = f32(h & 0xffu) / 255.0 * 6.0;
    let rgb = clamp(abs(fract(hue / 6.0 + vec3<f32>(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    let s = 0.5 + 0.5 * f32((h >> 8u) & 0xffu) / 255.0;
    return vec4<f32>(mix(vec3<f32>(1.0), rgb, s), params[0].y);
}
"#;

/// What one draw shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceParams {
    /// Maps the rectangle's NDC `(x, y, 0, 1)` to voxel coordinates, where
    /// voxel `i` spans `[i, i + 1)`.
    pub ndc_to_voxel: Matrix4,
    /// Handed to `slice_color` as `params`.
    pub params: [[f32; 4]; 4],
}

impl SliceParams {
    /// For [`WINDOW_LEVEL`]: `black` and `white` data values.
    pub fn window(ndc_to_voxel: Matrix4, black: f32, white: f32) -> Self {
        let mut params = [[0.0; 4]; 4];
        params[0] = [black, white, 0.0, 0.0];
        Self { ndc_to_voxel, params }
    }

    /// For [`LABELS`]: a colour `seed` and an `opacity`.
    pub fn labels(ndc_to_voxel: Matrix4, seed: u32, opacity: f32) -> Self {
        let mut params = [[0.0; 4]; 4];
        params[0] = [seed as f32, opacity, 0.0, 0.0];
        Self { ndc_to_voxel, params }
    }
}

/// The usual slice camera: maps a rectangle's NDC onto a plane through
/// `center` (voxels), turned by `orientation` (view axes to voxel axes), at
/// `voxels_per_pixel`, for a rectangle `viewport` pixels in size.
///
/// `y_down` puts view +y at the bottom of the rectangle, so image rows run
/// downward as they do in most volume data (neuroglancer's convention). With
/// it off, +y is up (three.js's).
pub fn plane_to_voxel(
    center: Vector3,
    orientation: Quaternion,
    voxels_per_pixel: f32,
    viewport: (f32, f32),
    y_down: bool,
) -> Matrix4 {
    let sy = if y_down { -1.0 } else { 1.0 };
    let half = Vector3::new(
        viewport.0 / 2.0 * voxels_per_pixel,
        sy * viewport.1 / 2.0 * voxels_per_pixel,
        1.0,
    );
    Matrix4::compose(center, orientation, half)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    ndc_to_voxel: [f32; 16],
    params: [[f32; 4]; 4],
    srgb_target: u32,
    _pad: [u32; 3],
}

const FRAGMENT: &str = r#"
struct SliceUniforms {
    ndc_to_voxel : mat4x4<f32>,
    params : array<vec4<f32>, 4>,
    srgb_target : u32,
};
@group(0) @binding(0) var<uniform> slice : SliceUniforms;
@group(0) @binding(1) var volume : texture_3d<SAMPLE>;

fn slice_srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs(in : FullscreenVertex) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let p = slice.ndc_to_voxel * vec4<f32>(ndc, 0.0, 1.0);
    let voxel = floor(p.xyz / p.w);
    let size = vec3<f32>(textureDimensions(volume));
    if (any(voxel < vec3<f32>(0.0)) || any(voxel >= size)) {
        discard;
    }
    var c = slice_color(textureLoad(volume, vec3<i32>(voxel), 0), slice.params);
    if (slice.srgb_target == 1u) {
        c = vec4<f32>(slice_srgb_to_linear(c.rgb), c.a);
    }
    return c;
}
"#;

/// How a [`SliceRenderer`] writes its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceTarget {
    pub format: wgpu::TextureFormat,
    /// `None` replaces what is there, alpha included.
    pub blend: Option<wgpu::BlendState>,
    /// A depth attachment to test and write. Every fragment is at depth 0, so
    /// with [`wgpu::CompareFunction::Less`] and a clear to 1, the first draw
    /// to reach a pixel keeps it: draw the most detailed data first and
    /// coarser data only fills the gaps.
    pub depth: Option<(wgpu::TextureFormat, wgpu::CompareFunction)>,
}

impl SliceTarget {
    /// Alpha blending over the target, no depth: what [`SliceRenderer::with_color`] makes.
    pub fn blended(format: wgpu::TextureFormat) -> Self {
        Self {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            depth: None,
        }
    }
}

/// The pipeline for one colour function, volume sample type and target
/// format. Make one, then a [`SliceBinding`] per rectangle you draw.
pub struct SliceRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sample: VolumeSample,
    srgb_target: bool,
    lookup: bool,
}

/// One rectangle's uniforms and volume.
///
/// Each rectangle drawn in a frame needs its own: uniforms written with
/// `Queue::write_buffer` all land before the frame's commands run, so draws
/// sharing a buffer would all see the last values written.
pub struct SliceBinding {
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SliceRenderer {
    /// Grey window/level over a float volume. See [`WINDOW_LEVEL`].
    pub fn window_level(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self::with_color(device, target_format, VolumeSample::Float, WINDOW_LEVEL)
    }

    /// A colour per label of an integer volume. See [`LABELS`].
    pub fn labels(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self::with_color(device, target_format, VolumeSample::Uint, LABELS)
    }

    /// Your own colour function: WGSL defining `slice_color` for `sample`
    /// (see the module docs).
    pub fn with_color(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample: VolumeSample,
        color_wgsl: &str,
    ) -> Self {
        Self::with_target(device, SliceTarget::blended(target_format), sample, color_wgsl)
    }

    /// [`with_color`](Self::with_color) with control over blending and
    /// depth; draw with [`draw_in_pass`](Self::draw_in_pass) into a pass
    /// whose attachments match `target`.
    pub fn with_target(
        device: &wgpu::Device,
        target: SliceTarget,
        sample: VolumeSample,
        color_wgsl: &str,
    ) -> Self {
        Self::build(device, target, sample, color_wgsl, false)
    }

    /// [`with_target`](Self::with_target), plus a lookup table the colour
    /// function can read: a `texture_2d<u32>` named `slice_lookup` (for
    /// example a sorted set of label IDs to search, or a colour map). Make
    /// bindings with [`create_binding_with_lookup`](Self::create_binding_with_lookup).
    pub fn with_lookup(device: &wgpu::Device, target: SliceTarget, sample: VolumeSample, color_wgsl: &str) -> Self {
        Self::build(device, target, sample, color_wgsl, true)
    }

    fn build(device: &wgpu::Device, target: SliceTarget, sample: VolumeSample, color_wgsl: &str, lookup: bool) -> Self {
        let target_format = target.format;
        let sample_type = match sample {
            VolumeSample::Float => wgpu::TextureSampleType::Float { filterable: false },
            VolumeSample::Uint => wgpu::TextureSampleType::Uint,
        };
        let lookup_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers slice layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                lookup_entry,
            ][..if lookup { 3 } else { 2 }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers slice pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers slice shader"),
            source: wgpu::ShaderSource::Wgsl(
                if lookup { Self::shader_source_with_lookup(sample, color_wgsl) } else { Self::shader_source(sample, color_wgsl) }.into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers slice pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(fullscreen::VS_ENTRY),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: target.blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: target.depth.map(|(format, compare)| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            sample,
            srgb_target: target_format.is_srgb(),
            lookup,
        }
    }

    /// The whole WGSL module [`with_color`](Self::with_color) compiles, with
    /// entry points `vs_main` and `fs`. For checking a colour function
    /// without a device, or reading what went wrong with one.
    /// [`shader_source`](Self::shader_source) for [`with_lookup`](Self::with_lookup).
    pub fn shader_source_with_lookup(sample: VolumeSample, color_wgsl: &str) -> String {
        format!("@group(0) @binding(2) var slice_lookup : texture_2d<u32>;\n{}", Self::shader_source(sample, color_wgsl))
    }

    pub fn shader_source(sample: VolumeSample, color_wgsl: &str) -> String {
        let scalar = match sample {
            VolumeSample::Float => "f32",
            VolumeSample::Uint => "u32",
        };
        format!(
            "{}\n{}\n{}",
            fullscreen::VS,
            color_wgsl,
            FRAGMENT.replace("SAMPLE", scalar)
        )
    }

    /// # Panics
    ///
    /// If the volume's sample type is not the one this renderer was made for.
    pub fn create_binding(&self, device: &wgpu::Device, volume: &GpuVolume) -> SliceBinding {
        assert!(!self.lookup, "this slice renderer reads a lookup table: use create_binding_with_lookup");
        self.binding(device, volume, None)
    }

    /// A binding for a [`with_lookup`](Self::with_lookup) renderer: `lookup`
    /// is a view of a 2D `Uint` texture (`R32Uint`, `Rg32Uint`, …).
    ///
    /// # Panics
    ///
    /// As [`create_binding`](Self::create_binding), and if this renderer
    /// takes no lookup table.
    pub fn create_binding_with_lookup(&self, device: &wgpu::Device, volume: &GpuVolume, lookup: &wgpu::TextureView) -> SliceBinding {
        assert!(self.lookup, "this slice renderer takes no lookup table: use create_binding");
        self.binding(device, volume, Some(lookup))
    }

    fn binding(&self, device: &wgpu::Device, volume: &GpuVolume, lookup: Option<&wgpu::TextureView>) -> SliceBinding {
        assert_eq!(
            volume.format.sample(),
            self.sample,
            "a {:?} volume cannot be drawn by a {:?} slice renderer",
            volume.format,
            self.sample
        );
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers slice uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&volume.view),
            },
        ];
        if let Some(view) = lookup {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers slice bind group"),
            layout: &self.layout,
            entries: &entries,
        });
        SliceBinding {
            uniforms,
            bind_group,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, binding: &SliceBinding, params: &SliceParams) {
        let uniforms = Uniforms {
            ndc_to_voxel: params.ndc_to_voxel.elements,
            params: params.params,
            srgb_target: self.srgb_target as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&binding.uniforms, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draws into `rect` of `target`, over what is already there.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        rect: PixelRect,
        binding: &SliceBinding,
    ) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        let mut pass = fullscreen::begin(encoder, target, fullscreen::Load::Keep);
        self.draw_in_pass(&mut pass, rect, binding);
    }

    /// Draws into `rect` within a pass you began, whose attachments match
    /// the [`SliceTarget`] this renderer was made for. Many draws can share
    /// one pass.
    pub fn draw_in_pass(&self, pass: &mut wgpu::RenderPass<'_>, rect: PixelRect, binding: &SliceBinding) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        pass.set_viewport(
            rect.x as f32,
            rect.y as f32,
            rect.width as f32,
            rect.height as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &binding.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(m: &Matrix4, x: f32, y: f32) -> Vector3 {
        Vector3::new(x, y, 0.0).apply_matrix4(m)
    }

    #[test]
    fn plane_to_voxel_puts_y_down_when_asked() {
        let center = Vector3::new(10.0, 20.0, 5.0);
        let m = plane_to_voxel(center, Quaternion::identity(), 0.5, (200.0, 100.0), true);
        assert_eq!(map(&m, 0.0, 0.0), center);
        assert_eq!(map(&m, 1.0, 0.0), Vector3::new(60.0, 20.0, 5.0));
        // NDC y = +1 is the top of the rectangle, which is lower y in the data.
        assert_eq!(map(&m, 0.0, 1.0), Vector3::new(10.0, -5.0, 5.0));

        let up = plane_to_voxel(center, Quaternion::identity(), 0.5, (200.0, 100.0), false);
        assert_eq!(map(&up, 0.0, 1.0), Vector3::new(10.0, 45.0, 5.0));
    }
}
