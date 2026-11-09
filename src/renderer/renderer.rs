//! wgpu scene renderer: pipelines, resource caches, shadows, and post-processing.
//!
//! Walks a [`Scene`](crate::Scene), batches draw calls by material and topology,
//! and writes to a swapchain view or [`super::RenderTarget`]. Post-fx passes
//! share WGSL in [`super::shader`] (`POSTFX_SHADER`, effect kinds 0–30).

use super::pipeline_state::Pipeline;
use crate::lights::shadow::{cascade_frusta, shadow_layer_size, Caster, MAX_SHADOW_RECORDS};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_mesh::{geom_cache_key, GpuMesh};
use super::gpu_texture::{
    cube_cache_key, tex_cache_key, GpuCubeTexture, GpuTexture, MipGenerator, TexCacheKey,
};
use super::gpu_timer::GpuTimer;
/// Identity of the texture set a bind group binds.
///
/// `external_rt_id` is folded in because RT-backed textures bypass the texture
/// cache entirely and resolve through `rt_view_cache`, so two textures with the
/// same upload key can still bind different views.
/// The second half is the morph texture bound at binding 10 — the geometry's
/// cache key and version, `(0, 0)` for the empty default.
/// The last part is the shared sampler's [`sampler_index`].
/// Then the geometry's uv-set texture ([`Renderer::uv_key`]).
type TexBgKey = (
    [Option<(TexCacheKey, u32)>; 8],
    (usize, u32),
    usize,
    (usize, u32),
);

/// A material variant's group 2: the variant, then each of its maps' upload
/// key, render-target id, sampler state and uv transform (bit patterns), then
/// the morph texture, then `Material::map_uniforms` and the `envMap`'s identity.
type VariantBgKey = (
    super::material_maps::MapVariant,
    Vec<(TexCacheKey, u32, u32, [u32; 6])>,
    (usize, u32),
    [[u32; 4]; 6],
    usize,
);

/// One draw of a mesh: its material, and the index range it covers (`None`
/// for all of it), as three.js's render list holds a mesh with groups.
type DrawItem<M> = (M, Option<(u32, u32)>);

fn variant_bg_key(
    slots: &MaterialTextureSlots,
    v: super::material_maps::MapVariant,
    morph: (usize, u32),
    uniforms: [[f32; 4]; 6],
) -> VariantBgKey {
    let maps = super::material_maps::role_textures(slots)
        .into_iter()
        .filter(|(r, _)| v.contains(*r))
        .filter_map(|(_, t)| t)
        .map(|t| {
            let sampler = sampler_index(t) as u32;
            let [r0, r1] = t.uv_transform_rows();
            let xform = [
                r0[0].to_bits(),
                r0[1].to_bits(),
                r0[2].to_bits(),
                r1[0].to_bits(),
                r1[1].to_bits(),
                r1[2].to_bits(),
            ];
            (
                tex_cache_key(t),
                t.external_rt_id.unwrap_or(u32::MAX),
                sampler,
                xform,
            )
        })
        .collect();
    let env = slots
        .env_map
        .as_ref()
        .map_or(0, |c| std::sync::Arc::as_ptr(c) as usize);
    (v, maps, morph, uniforms.map(|r| r.map(f32::to_bits)), env)
}

fn tex_slot_key(slot: &Option<std::sync::Arc<crate::Texture>>) -> Option<(TexCacheKey, u32)> {
    slot.as_ref()
        .map(|t| (tex_cache_key(t), t.external_rt_id.unwrap_or(u32::MAX)))
}

fn tex_bg_key(slots: &MaterialTextureSlots) -> TexBgKey {
    (
        [
            tex_slot_key(&slots.map),
            tex_slot_key(&slots.normal_map),
            tex_slot_key(&slots.roughness_map),
            tex_slot_key(&slots.metalness_map),
            tex_slot_key(&slots.ao_map),
            tex_slot_key(&slots.emissive_map),
            tex_slot_key(&slots.matcap_map),
            tex_slot_key(&slots.cloud_shadow_map),
        ],
        (0, 0),
        slots
            .sampler_source()
            .map_or(SAMPLER_DEFAULT, |t| sampler_index(t)),
        (0, 0),
    )
}

/// Every combination of filter and wrap a texture can ask for: mag × min
/// filter (2 × 2) × `wrapS` × `wrapT` (3 × 3) × mipmap mode (3: linear,
/// nearest, none). three.js gives each texture its own sampler state, so a
/// texture's sampler is looked up, never approximated.
const SAMPLER_COUNT: usize = 108;

/// A map-less material's sampler: linear, trilinear, clamped — three.js's
/// defaults for a `DataTexture`.
const SAMPLER_DEFAULT: usize = 0;

/// `mipmap_filter` as a sampler-table digit: 0 linear (three.js's default),
/// 1 nearest, 2 none.
fn mip_code(m: Option<crate::textures::TextureFilter>) -> usize {
    match m {
        Some(crate::textures::TextureFilter::Linear) => 0,
        Some(crate::textures::TextureFilter::Nearest) => 1,
        None => 2,
    }
}

/// Index into `Renderer::samplers` for `t`'s `mag_filter`, `min_filter`,
/// `mipmap_filter`, `wrap_s` and `wrap_t`.
fn sampler_index(t: &Texture) -> usize {
    use crate::textures::{TextureFilter, TextureWrap};
    let wrap = |w: TextureWrap| match w {
        TextureWrap::ClampToEdge => 0,
        TextureWrap::Repeat => 1,
        TextureWrap::MirroredRepeat => 2,
    };
    (t.mag_filter == TextureFilter::Nearest) as usize
        + ((t.min_filter == TextureFilter::Nearest) as usize) * 2
        + wrap(t.wrap_s) * 4
        + wrap(t.wrap_t) * 12
        + mip_code(t.mipmap_filter) * 36
}

/// [`sampler_index`] for a cube texture: its filters, clamped — a cube map's
/// seams are its faces' edges.
fn cube_sampler_index(c: &crate::textures::CubeTexture) -> usize {
    use crate::textures::TextureFilter;
    (c.mag_filter == TextureFilter::Nearest) as usize
        + ((c.min_filter == TextureFilter::Nearest) as usize) * 2
        + mip_code(c.mipmap_filter) * 36
}

/// The sampler [`sampler_index`] `i` names. With no mipmaps the level of detail
/// is held at 0, so minification samples the full-size image as three.js's
/// `LinearFilter` / `NearestFilter` do; anisotropic filtering only where every
/// filter is linear (wgpu allows it nowhere else).
fn make_sampler(device: &wgpu::Device, i: usize) -> wgpu::Sampler {
    let filter = |nearest: bool| {
        if nearest {
            wgpu::FilterMode::Nearest
        } else {
            wgpu::FilterMode::Linear
        }
    };
    let address = |w: usize| match w {
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    };
    let (mag_nearest, min_nearest) = (i & 1 != 0, i & 2 != 0);
    let mip = i / 36;
    let all_linear = !mag_nearest && !min_nearest && mip == 0;
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("threers map sampler"),
        address_mode_u: address((i / 4) % 3),
        address_mode_v: address((i / 12) % 3),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filter(mag_nearest),
        min_filter: filter(min_nearest),
        mipmap_filter: if mip == 0 {
            wgpu::MipmapFilterMode::Linear
        } else {
            wgpu::MipmapFilterMode::Nearest
        },
        lod_max_clamp: if mip == 2 { 0.0 } else { 32.0 },
        anisotropy_clamp: if all_linear { 16 } else { 1 },
        ..Default::default()
    })
}

/// A draw's index (or vertex) range: its span, clamped to `count`, or all of
/// it.
/// Make `current` the `w × h` pair if it or `spare` already is (swapping
/// them), and say whether a new one has to be made. When it does, the old
/// current one is kept as the spare.
fn take_sized<T>(
    current: &mut Option<(u32, u32, T, T)>,
    spare: &mut Option<(u32, u32, T, T)>,
    w: u32,
    h: u32,
) -> bool {
    let fits =
        |t: &Option<(u32, u32, T, T)>| matches!(t, Some((cw, ch, ..)) if *cw == w && *ch == h);
    if fits(current) {
        return false;
    }
    std::mem::swap(current, spare);
    !fits(current)
}

fn draw_span(range: Option<(u32, u32)>, count: u32) -> std::ops::Range<u32> {
    match range {
        Some((start, end)) => start.min(count)..end.min(count),
        None => 0..count,
    }
}

/// One geometry's morph targets on the GPU: three.js's morph texture, an
/// `Rgba32Float` 2D array with a layer per target (see
/// [`crate::core::MorphTexels`]).
/// three.js's transmission render target (`transmissionRenderTarget`): the
/// opaque scene drawn linear and untone-mapped at the main target's size, with
/// a full mip chain the transmissive materials read blurred by roughness.
struct TransmissionCapture {
    texture: wgpu::Texture,
    /// Every level, for sampling.
    view: wgpu::TextureView,
    /// Level 0, for drawing into.
    target_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// A copy the back-face pass samples while it draws into `texture`.
    copy: wgpu::Texture,
    copy_view: wgpu::TextureView,
    size: (u32, u32),
    levels: u32,
}

/// A geometry's extra uv sets — three.js's `uv1`, `uv2` and `uv3` — as one
/// texture the vertex stage reads by vertex index (`vertex_uv_sets` in the
/// shader): two `Rgba32Float` texels a vertex, `(uv1, uv2)` and `(uv3, 0, 0)`,
/// row-major. A texture rather than vertex attributes so the interleaved
/// vertex layout [`Renderer::vertex_buffer`] publishes stays as it is, and so
/// no pipeline grows a vertex buffer that most geometry would leave unbound.
struct UvSetTexture {
    version: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// The three.js names of the extra uv sets, in channel order from 1.
const UV_SET_NAMES: [&str; 3] = ["uv1", "uv2", "uv3"];

/// The group-2 layout entry for [`UvSetTexture`] (binding 8), which every
/// layout the shared vertex stage runs under declares.
fn uv_sets_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 8,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

struct MorphTexture {
    version: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// For an `InstancedMesh` with a morph texture: the layer holding the
    /// per-instance rows and the texture's width, so the rows can be
    /// rewritten in place when only the influences change.
    instance_layer: Option<(u32, u32)>,
}

use super::shader::{
    MAX_DIR_LIGHTS, MAX_HEMI_LIGHTS, MAX_POINT_LIGHTS, MAX_RECT_LIGHTS, MAX_SPOT_LIGHTS,
};
use crate::cameras::Camera;
use crate::core::{BufferGeometry, ObjectKind};
use crate::lights::Light;
use crate::materials::{Material, MaterialKind, MaterialTextureSlots};
use crate::math::{Frustum, Matrix3, Matrix4, Sphere, Vector3};
use crate::scene::Scene;
use crate::textures::{CubeTexture, Texture, TextureFormat};

// Post-fx draws described as data, for the composer in `crate::postprocessing`.
mod postfx_op;
pub(crate) use postfx_op::stored_clear;
pub use postfx_op::{PostFxBlend, PostFxNoise, PostFxOp, PostFxOutput};

fn rt_sample_format(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    if format == wgpu::TextureFormat::Rgba16Float {
        format
    } else {
        format.add_srgb_suffix()
    }
}

// ---------- GPU layout types ----------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct DirLightGpu {
    direction: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct PointLightGpu {
    position: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct SpotLightGpu {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct HemiLightGpu {
    sky_color: [f32; 4],
    ground_color: [f32; 4],
    direction: [f32; 4],
}

/// A rectangular area light. `right`/`up` are **half-extent** vectors in world
/// space, so the four corners are `position ± right ± up` and the emitting face
/// normal is `normalize(cross(right, up))`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct RectLightGpu {
    position: [f32; 4],
    color: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

/// Group 3's `EnvUniforms` (binding 13): the environment one draw's
/// image-based light reads — the scene's or a Standard/Physical material's own
/// `envMap` — with its sampling parameters, intensity and rotation. See the
/// WGSL struct of the same name.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvUniforms {
    /// PMREM atlas texel width and height, its top mip (or the plain cube's
    /// last mip), and 1 when it is an atlas.
    params: [f32; 4],
    /// x: 1 when there is an environment; y: its intensity.
    info: [f32; 4],
    /// Rows of the lookup rotation.
    rot: [[f32; 4]; 3],
}

impl EnvUniforms {
    /// No environment: the image-based light is off.
    const NONE: Self = Self {
        params: [0.0; 4],
        info: [0.0, 1.0, 0.0, 0.0],
        rot: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    /// `cube`, uploaded as `gpu`, at `intensity` and `rotation`.
    fn of(
        cube: &CubeTexture,
        gpu: &GpuCubeTexture,
        intensity: f32,
        rotation: crate::math::Euler,
    ) -> Self {
        let params = match &cube.cube_uv_atlas {
            Some(a) => [a.texel_width, a.texel_height, a.lod_max as f32, 1.0],
            None => [
                0.0,
                0.0,
                gpu.texture.mip_level_count().saturating_sub(1) as f32,
                0.0,
            ],
        };
        Self {
            params,
            info: [1.0, intensity, 0.0, 0.0],
            rot: crate::materials::environment_lookup_rows(rotation, cube.cube_uv_atlas.is_none()),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view: [f32; 16],
    projection: [f32; 16],
    view_proj: [f32; 16],
    camera_position: [f32; 4],
    ambient: [f32; 4],
    light_counts: [u32; 4],
    /// x: tone-mapping mode (0 none, 1 linear, 2 ACES), y: exposure, z: IBL on,
    /// w: linear framebuffer (1 = HalfFloat RT, skip sRGB encode on write)
    tone_mapping_exposure: [f32; 4],
    /// Fog color rgb + a (a unused). Multiplied with the mix factor in shader.
    fog_color: [f32; 4],
    /// x: near (linear), y: far (linear), z: density (exp2), w: mode (0/1/2)
    fog_params: [f32; 4],
    /// x: viewport width (pixels), y: viewport height (pixels), zw: reserved.
    viewport_size: [f32; 4],
    /// CubeUV PMREM: x/y texel size, z max mip, w = 1 when CubeUV active.
    /// The shader reads these from `EnvUniforms` now, per environment; kept
    /// here for the custom-shader preamble's `Frame` layout.
    env_map_params: [f32; 4],
    dir_lights: [DirLightGpu; MAX_DIR_LIGHTS],
    point_lights: [PointLightGpu; MAX_POINT_LIGHTS],
    spot_lights: [SpotLightGpu; MAX_SPOT_LIGHTS],
    hemi_lights: [HemiLightGpu; MAX_HEMI_LIGHTS],
    rect_lights: [RectLightGpu; MAX_RECT_LIGHTS],
    /// x: rect-area light count, y: geometric specular antialiasing on,
    /// z: reflective transparency on, w: the transmission capture's mip count
    /// (0 = none). `light_counts` was already full.
    light_counts2: [u32; 4],
    /// x: shadow records in use.
    shadow_info: [u32; 4],
    /// One per shadow-casting light, which names it in its own `color[3]`
    /// (record + 1). See `ShadowRecordGpu`.
    shadows: [ShadowRecordGpu; MAX_SHADOW_RECORDS],
}

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            view: [0.0; 16],
            projection: [0.0; 16],
            view_proj: [0.0; 16],
            camera_position: [0.0; 4],
            ambient: [0.0; 4],
            light_counts: [0; 4],
            // Default: NoToneMapping, exposure 1.0 — matches three.js default.
            tone_mapping_exposure: [0.0, 1.0, 0.0, 0.0],
            fog_color: [1.0, 1.0, 1.0, 1.0],
            fog_params: [1.0, 1000.0, 0.0, 0.0],
            viewport_size: [1.0, 1.0, 0.0, 0.0],
            env_map_params: [0.0; 4],
            dir_lights: [DirLightGpu::default(); MAX_DIR_LIGHTS],
            point_lights: [PointLightGpu::default(); MAX_POINT_LIGHTS],
            spot_lights: [SpotLightGpu::default(); MAX_SPOT_LIGHTS],
            hemi_lights: [HemiLightGpu::default(); MAX_HEMI_LIGHTS],
            rect_lights: [RectLightGpu::default(); MAX_RECT_LIGHTS],
            light_counts2: [0; 4],
            shadow_info: [0; 4],
            shadows: [ShadowRecordGpu::zeroed(); MAX_SHADOW_RECORDS],
        }
    }
}

/// Build the environment `@group(3)` bind group. The layout (13 bindings: env
/// cube + samplers, three shadow maps + comparison samplers, the CubeUV atlas,
/// and the screen-space glass color/depth/back-depth) is fixed here in one place,
/// so the four construction sites (default, shadow-dummies, per-frame, glass)
/// just pass the 13 views/samplers in binding order and a new binding is added
/// once. Texture vs. sampler wrapping is encoded by the arg position.
/// The environment/shadow bind group.
///
/// The screen-space capture is deliberately not here. It belongs to the
/// refraction pass's own group 3, built by hand at its call site, because a
/// declared binding costs a texture unit whether or not the entry point reads it
/// — so carrying it here would have cost every pipeline three units it never
/// uses, which is one more than WebGL2 has to spare.
macro_rules! env_bind_group {
    ($device:expr, $bgl:expr, $label:expr,
     $t0:expr, $s1:expr, $t2:expr, $s3:expr, $t4:expr, $s5:expr,
     $t6:expr, $s7:expr, $t8:expr, $u13:expr, $ltc14:expr $(,)?) => {{
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView($t0),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler($s1),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView($t2),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler($s3),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView($t4),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler($s5),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView($t6),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler($s7),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView($t8),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: $u13.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: wgpu::BindingResource::TextureView($ltc14),
            },
        ];
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $bgl,
            entries: &entries,
        })
    }};
}

/// The custom (`ShaderMaterial`) pipeline's trimmed group 3, where the backend
/// cannot hold the full environment beside the volume slots (WebGL2): the IBL
/// cube (0), its sampler (1), the 2D cube-UV fallback (8) and the environment
/// uniform (13) — what the preamble's image-based-light helpers read — and
/// none of the shadow maps or the area-light table, which a custom fragment
/// does its own lighting without. Matches `env_bgl` filtered to those bindings.
macro_rules! custom_env_bind_group {
    ($device:expr, $bgl:expr, $label:expr, $t0:expr, $s1:expr, $t8:expr, $u13:expr $(,)?) => {{
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView($t0),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler($s1),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView($t8),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: $u13.as_entire_binding(),
            },
        ];
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $bgl,
            entries: &entries,
        })
    }};
}

/// A material's own environment: the cube's upload, its intensity and its
/// rotation (angle bit patterns, then the axis order's index).
type MaterialEnvKey = (super::gpu_texture::CubeCacheKey, u32, [u32; 4]);

fn material_env_key(
    env: (&std::sync::Arc<CubeTexture>, f32, crate::math::Euler),
) -> MaterialEnvKey {
    let (cube, intensity, r) = env;
    let order = crate::math::EulerOrder::ALL
        .iter()
        .position(|o| *o == r.order)
        .unwrap_or(0) as u32;
    (
        cube_cache_key(cube),
        intensity.to_bits(),
        [r.x.to_bits(), r.y.to_bits(), r.z.to_bits(), order],
    )
}

/// Views and bind groups the temporal pass reuses across an accumulation run.
///
/// Keyed by nothing: it is dropped whenever `ensure_taa_targets` rebuilds the
/// history, which is the only thing any of it depends on.
struct TaaBindCache {
    /// Resolve bind groups, indexed by `taa_frame % 2` — the history buffers
    /// ping-pong, so there are exactly two arrangements.
    resolve: [wgpu::BindGroup; 2],
    /// Blit bind groups, same indexing.
    blit: [wgpu::BindGroup; 2],
}

/// One cached instance buffer: see `Renderer::instance_buffer`.
struct InstanceBuffer {
    buffer: wgpu::Buffer,
    /// What it holds, to skip rewriting unchanged data.
    data: Vec<f32>,
    /// The `instance_frame` it was last used in.
    frame: u64,
    /// The mesh's `instance_version` and instance count when written, for
    /// a mesh that declares one.
    version: Option<(u64, usize)>,
    /// Changes whenever what it holds does: what the shadow fingerprint
    /// reads to tell a still instanced caster from a moving one.
    stamp: u64,
}

/// Uniform for the TAA resolve pass — see `TAA_SHADER`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct TaaUniforms {
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    /// x: width, y: height, z: history weight, w: first-frame flag.
    params: [f32; 4],
    /// xyz: how far the camera moved since the previous frame. w: converge
    /// mode — 0 for the feedback blend, 1 for a running mean over
    /// sRGB-encoded bytes, 2 for a running mean over linear light.
    cam_delta: [f32; 4],
    /// x: 1 when the depth it reads is reversed (`set_reversed_z`).
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct MeshUniforms {
    model: [f32; 16],
    normal_matrix: [f32; 16],
    color: [f32; 4],
    emissive: [f32; 4],
    specular: [f32; 4],
    /// x: shininess/depth_near, y: opacity/depth_far, z: roughness, w: metalness
    params: [f32; 4],
    /// x: ao_intensity, y/z: normal_scale, w: unused
    params2: [f32; 4],
    /// Physical layers: x: clearcoat, y: clearcoat_roughness, z: ior, w: transmission
    params3: [f32; 4],
    /// Physical volume: x: thickness (glass refraction/march depth),
    /// y: dispersion, z: vertex_emissive, w: anisotropy_rotation (radians).
    params4: [f32; 4],
    /// Physical layers 2: x: iridescence, y: iridescence_ior,
    /// z: iridescence_thickness (nm), w: sheen strength.
    params5: [f32; 4],
    /// x/y/z: sheen_color, w: sheen_roughness.
    params6: [f32; 4],
    /// x/y/z: attenuation_color, w: attenuation_distance (0 = disabled, which
    /// keeps the legacy "tint transmission by base color" behaviour).
    params7: [f32; 4],
    /// x: displacement_scale, y: displacement_bias, z: emissive_night_side
    /// (w reserved).
    params8: [f32; 4],
    /// UV transform from the primary map: xy offset, zw repeat.
    params9: [f32; 4],
    /// x: cloud shell height, y: cloud shadow strength, z: cloud longitude
    /// offset (turns), w: twilight wrap width.
    params10: [f32; 4],
    /// xyz: twilight colour, w: the sun's angular radius in radians.
    params11: [f32; 4],
    /// Eclipse occluder: world-space xyz centre, w radius (0 = none).
    params12: [f32; 4],
    /// x: cloud forward-scatter strength, y: scattering anisotropy g,
    /// z: ozone absorption strength, w: aurora strength.
    params13: [f32; 4],
    /// xyz: aurora colour, w: geomagnetic colatitude of the auroral oval, in
    /// degrees from the pole.
    params14: [f32; 4],
    /// x: material kind, y: texture flags
    flags: [u32; 4],
    /// Morph targets: (count, base influence, texels per vertex, flags) — see
    /// `MeshUniforms.morph` in the shader and [`crate::core::morph_uniforms`].
    morph: [f32; 4],
    /// `morphTargetInfluences`, zero-padded to `MAX_MORPH_TARGETS`, four to
    /// a vector.
    morph_weights: [[f32; 4]; crate::core::MAX_MORPH_TARGETS / 4],
    /// `Object3D::clip_plane` as (normal, constant); all zero clips nothing.
    clip_plane: [f32; 4],
}

/// `Object3D::clip_plane` for `MeshUniforms`: the shader treats an all-zero
/// plane as none (no real plane has a zero normal).
fn clip_plane_of(obj: &crate::core::Object3D) -> [f32; 4] {
    obj.clip_plane.map_or([0.0; 4], |p| {
        [p.normal.x, p.normal.y, p.normal.z, p.constant]
    })
}

// Texture-flag bits — keep in sync with shader.rs FLAG_*.
const FLAG_MAP: u32 = 1;
const FLAG_NORMAL_MAP: u32 = 2;
const FLAG_ROUGHNESS_MAP: u32 = 4;
const FLAG_METALNESS_MAP: u32 = 8;
const FLAG_AO_MAP: u32 = 16;
const FLAG_EMISSIVE_MAP: u32 = 32;
const FLAG_MATCAP_MAP: u32 = 64;
const FLAG_DASHED: u32 = 128;
const FLAG_SHADOW_MAT: u32 = 256;
const FLAG_RECEIVE_SHADOW: u32 = 512;
const FLAG_CLOUD_SHADOW: u32 = 4096;
/// The draw writes into a plain (`Unorm`) view of an sRGB target, so the shader
/// encodes its output itself — the encoded-blend pass, see `encoded_pass`.
const FLAG_ENCODED_OUT: u32 = 8192;
/// three.js `toneMapped: false` (see `apply_tone_mapping` in the shader).
const FLAG_NO_TONE_MAP: u32 = 16384;
/// `mesh.flags.z`: a sprite or point drawn on the blended pipeline whose
/// material is not `transparent` — written with alpha 1, which blends to a
/// plain write, as three.js draws an opaque sprite.
const FLAG_OPAQUE_OUT: u32 = 32768;
/// `mesh.flags.z`: a transmissive `PhysicalMaterial` in `TransparencyMode::Glass`
/// or `Oit`. Those modes blend, so the background arrives through the blend
/// rather than from the transmission capture, which is only made for
/// three.js's own transmission.
const FLAG_BLENDED_GLASS: u32 = 65536;
/// `mesh.flags.z`: thin transmissive glass — a `PhysicalMaterial` with
/// `transmission` and either `thickness` 0 (three.js's thin-walled) or
/// `TransparencyMode::Glass`. Light goes straight through it, so its shadow
/// casting drops the share of shadow-map texels that gets through and the
/// shadow filter averages it into a partial shadow (`shadow_cut` in the
/// shader). Solid glass bends that light into a caustic instead, and keeps a
/// full shadow.
const FLAG_SHADOW_TRANSMITS: u32 = 131072;

/// Shadow pipelines for casters cut by `alphaTest` and a map (three.js's
/// shadow depth material takes both): `fs_shadow_cut` discards where the map's
/// alpha is under the test.
/// `base` for a draw on material variant `v` — itself for `None`. A free
/// function over the renderer's fields, because it is called inside render
/// passes that hold other borrows of the renderer. The variant must already be
/// built (`Renderer::ensure_map_variant`, done for every draw before the
/// passes).
/// `m` remapped to reversed depth: clip `z' = w - z`, so NDC depth runs from
/// 1 at the near plane to 0 at the far one, with x, y and w untouched.
fn reverse_depth(m: &crate::math::Matrix4) -> crate::math::Matrix4 {
    let mut out = *m;
    for c in 0..4 {
        out.elements[c * 4 + 2] = m.elements[c * 4 + 3] - m.elements[c * 4 + 2];
    }
    out
}

fn variant_pipeline(
    pipes: &mut super::pipeline_state::StatePipelines,
    variants: &HashMap<
        super::material_maps::MapVariant,
        (wgpu::ShaderModule, wgpu::BindGroupLayout),
    >,
    device: &wgpu::Device,
    main: &wgpu::ShaderModule,
    base: &Pipeline,
    v: Option<super::material_maps::MapVariant>,
) -> Pipeline {
    let Some(v) = v else {
        return base.clone();
    };
    let Some((module, bgl)) = variants.get(&v) else {
        log::error!("threers: a draw names material variant {v:?}, which was not built");
        return base.clone();
    };
    pipes
        .relayout(device, base, v.key(), 2, bgl, main, module)
        .unwrap_or_else(|| {
            // Every pipeline that binds a material's group 2 is recorded, so this
            // is a bug: the bind group will not match the pipeline. Say so.
            log::error!("threers: a material-variant draw reached an unrecorded pipeline");
            base.clone()
        })
}

/// three.js `WebGLShadowMap`'s `shadowSide` for a material `side`: a
/// front-sided caster records its back faces, a back-sided one its front
/// faces, and a double-sided one both.
/// three.js `side` for a mesh drawn as faces in the colour passes: FrontSide
/// culls back faces, BackSide front faces, DoubleSide none. `None` for what is
/// not faces (lines, points, sprites, the sky).
fn side_cull(faces: bool, side: u32) -> Option<Option<wgpu::Face>> {
    faces.then_some(match side {
        1 => Some(wgpu::Face::Front),
        2 => None,
        _ => Some(wgpu::Face::Back),
    })
}

fn shadow_cull(side: u32) -> Option<wgpu::Face> {
    match side {
        1 => Some(wgpu::Face::Back),
        2 => None,
        _ => Some(wgpu::Face::Front),
    }
}

/// [`shadow_cull`] for the point-light pass, whose cube faces negate clip y
/// (cube maps are left-handed): that one mirror reverses every triangle's
/// winding, so the faces to cull swap.
fn shadow_cull_point(side: u32) -> Option<wgpu::Face> {
    match shadow_cull(side) {
        Some(wgpu::Face::Front) => Some(wgpu::Face::Back),
        Some(wgpu::Face::Back) => Some(wgpu::Face::Front),
        None => None,
    }
}

struct ShadowCutPipelines {
    dir: Pipeline,
    spot: Pipeline,
    dir_skinned: Pipeline,
    spot_skinned: Pipeline,
    dir_instanced: Pipeline,
    spot_instanced: Pipeline,
}

struct CachedGpuMesh {
    version: u32,
    mesh: GpuMesh,
    /// Object-space bounds, for frustum culling. `None` when the geometry has
    /// no position attribute, which is the one case that must never be culled
    /// — there is nothing to test it against.
    sphere: Option<Sphere>,
}

/// A bounding sphere for `geom` without mutating it.
///
/// `BufferGeometry::compute_bounding_sphere` caches into the geometry and so
/// needs `&mut`; geometries reach the renderer behind an `Arc` and are shared,
/// so this recomputes instead. Same construction: centre of the AABB, radius
/// the furthest vertex from it.
fn local_bounding_sphere(geom: &BufferGeometry) -> Option<Sphere> {
    if let Some(s) = geom.bounding_sphere {
        return Some(s);
    }
    // A compute pass moves its vertices; the CPU positions are not where
    // it is, so it has no bounds to cull by.
    if geom.gpu_writable {
        return None;
    }
    let mut lo = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut hi = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in geom.positions()? {
        lo = Vector3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vector3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    if !lo.x.is_finite() {
        return None;
    }
    let centre = (lo + hi) * 0.5;
    let mut r2 = 0.0f32;
    for p in geom.positions()? {
        r2 = r2.max((p - centre).length_sq());
    }
    Some(Sphere::new(centre, r2.sqrt()))
}

/// Camera parameters forwarded into SSAO / SSR post-fx uniforms.
#[derive(Clone, Debug)]
pub struct PostFxCamera {
    pub near: f32,
    pub far: f32,
    pub kernel_radius: f32,
    pub kernel_size: u32,
    pub proj: [f32; 16],
    pub inv_proj: [f32; 16],
}

impl Default for PostFxCamera {
    fn default() -> Self {
        let mut id = [0f32; 16];
        id[0] = 1.0;
        id[5] = 1.0;
        id[10] = 1.0;
        id[15] = 1.0;
        Self {
            near: 0.1,
            far: 100.0,
            kernel_radius: 8.0,
            kernel_size: 32,
            proj: id,
            inv_proj: id,
        }
    }
}

/// Tone mapping lives in [`crate::math`], so the renderers without a GPU have it too.
pub use crate::math::tone_mapping::ToneMapping;

mod construct;
mod frame;
mod geometry;
mod postfx;
mod targets;
mod textures;

/// Everything the custom pipeline descriptor branches on. Anything left out of
/// this is a material silently drawing with another material's pipeline.
/// The material's `RenderState` is not in it: that is applied over the stock
/// pipeline as a state variant (see `pipeline_state`).
type CustomPipelineKey = (u64, bool, bool, bool, u32);

/// A cache keyed by a geometry's address.
///
/// The address is an identity and nothing more: no key is ever dereferenced
/// (the impl below is the only `unsafe` in this file), so the map can move
/// between threads with the renderer that owns it. A raw pointer in the key
/// would otherwise make the whole [`Renderer`] `!Send`, and a renderer
/// drawing into an app's layer has to be handed to the thread that draws.
struct IdentityKeyed<K, V>(HashMap<K, V>);

// SAFETY: see above — the pointers in `K` are compared and hashed, never
// followed. `V` carries its own bound.
unsafe impl<K, V: Send> Send for IdentityKeyed<K, V> {}

impl<K, V> Default for IdentityKeyed<K, V> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<K, V> std::ops::Deref for IdentityKeyed<K, V> {
    type Target = HashMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K, V> std::ops::DerefMut for IdentityKeyed<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// GPU renderer: scene draws, shadow maps, environment sampling, post-fx.
/// How a [`Renderer`] builds its pipelines — see [`Renderer::with_options`].
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct RendererOptions {
    /// Build the initial pipelines on first use rather than in the
    /// constructor ([`Renderer::new_lazy`]). [`Renderer::warm_up`] builds
    /// them ahead of time, on any thread.
    pub lazy: bool,
    /// Build every pipeline the renderer makes through this cache. Load one
    /// from wherever the app keeps it and save it after a warm-up, and the
    /// next run skips the driver's shader compilation — see
    /// [`PipelineCache`](crate::renderer::PipelineCache). Needs
    /// `Features::PIPELINE_CACHE` (Vulkan); `None` elsewhere.
    pub pipeline_cache: Option<wgpu::PipelineCache>,
}

impl RendererOptions {
    /// Initial pipelines on first use.
    pub fn lazy(mut self, lazy: bool) -> Self {
        self.lazy = lazy;
        self
    }

    /// Pipelines through `cache`: a threers
    /// [`PipelineCache`](crate::renderer::PipelineCache) (`&cache`, or the
    /// `Option` its `open` returns, as it comes) or a `wgpu::PipelineCache`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pipeline_cache(
        mut self,
        cache: impl crate::renderer::pipeline_cache::IntoPipelineCache,
    ) -> Self {
        self.pipeline_cache = cache.into_pipeline_cache();
        self
    }
}

/// Frames a texture nothing draws keeps its GPU copy, by default — see
/// [`Renderer::set_cache_retention`]. Two seconds at 60 fps: long enough for a
/// texture shown on and off, short enough that a stream of new ones does not
/// pile up.
pub const DEFAULT_CACHE_RETENTION: u64 = 120;

pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_tri: Pipeline,
    pipeline_tri_alpha: Pipeline,
    // Single-layer glass (depth prepass + LessEqual color pass), sRGB + f16.
    pipeline_glass_depth: Pipeline,
    pipeline_glass_color: Pipeline,
    pipeline_glass_depth_f16: Pipeline,
    pipeline_glass_color_f16: Pipeline,
    pipeline_glass_depth_msaa: Pipeline,
    pipeline_glass_color_msaa: Pipeline,
    /// The tint half of the glass colour pass (`GLASS_OUTPUT` 1), drawn just
    /// before the colour half: what is behind times the glass's per-channel
    /// transmittance.
    pipeline_glass_tint: Pipeline,
    pipeline_glass_tint_f16: Pipeline,
    pipeline_glass_tint_msaa: Pipeline,
    /// `[depth, color, tint]` glass for the encoded-blend pass under MSAA
    /// (sRGB targets only, as `encoded_blend_pipelines`).
    encoded_glass_msaa: Option<[Pipeline; 3]>,
    // Weighted-blended OIT: accumulation pipeline (MRT) + resolve (sRGB + f16).
    /// Order-independent transparency accumulation: one pipeline writing both
    /// attachments, or where attachments cannot blend independently (see
    /// [`caps::supports_independent_blend`](crate::renderer::caps::supports_independent_blend))
    /// one each, drawn in a pass of its own.
    pipeline_oit: Vec<Pipeline>,
    /// `[tint, light]`: the OIT composite's two halves.
    pipeline_oit_resolve: [Pipeline; 2],
    pipeline_oit_resolve_f16: [Pipeline; 2],
    pipeline_oit_resolve_msaa: [Pipeline; 2],
    /// OIT accumulation under MSAA, and its multisampled `(width, height,
    /// accum, reveal)`, resolved into `oit_targets` at the end of the pass.
    pipeline_oit_msaa: Vec<Pipeline>,
    oit_msaa_targets: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// As `msaa_targets_spare`, for these.
    oit_msaa_targets_spare: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// The multisampled opaque depth to the single-sampled one, for the
    /// refraction pass under MSAA (`DEPTH_RESOLVE_SHADER`). `None` where a
    /// shader cannot read a multisampled texture.
    depth_resolve: Option<(wgpu::BindGroupLayout, Pipeline)>,
    /// Where depth textures cannot be copied (WebGL2): a fullscreen pass that
    /// samples one depth and writes it as another's, with its layout and the
    /// nearest sampler it reads through.
    depth_copy: Option<(wgpu::BindGroupLayout, Pipeline, wgpu::Sampler)>,
    oit_resolve_bgl: wgpu::BindGroupLayout,
    // accum(Rgba16F) + revealage(R16F) targets + the resolve bind group, keyed by size.
    oit_targets: Option<(
        u32,
        u32,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::BindGroup,
    )>,
    /// No-cull triangle pipeline, used for `side: BackSide` / `side: DoubleSide`
    /// materials. Without this, three.js apps using BackSide for skybox-style
    /// "inside of a box" rendering silently get culled-to-nothing.
    pipeline_tri_nocull: Pipeline,
    /// Sky dome: no cull, no depth write, LessEqual (matches three.js Sky depthWrite:false).
    pipeline_tri_sky: Pipeline,
    pipeline_tri_wire: Pipeline,
    pipeline_line: Pipeline,
    pipeline_point: Pipeline,
    pipeline_point_alpha: Pipeline,
    /// `LineSegments` wider than a pixel (`vs_wide_lines`), opaque and blended.
    pipeline_wide_line: Pipeline,
    pipeline_wide_line_alpha: Pipeline,
    pipeline_sprite: Pipeline,
    // Hardware-MSAA variants of the opaque forward pipelines (surface path).
    pipeline_tri_msaa: Pipeline,
    pipeline_skinned_msaa: Pipeline,
    pipeline_instanced_msaa: Pipeline,
    pipeline_tri_alpha_msaa: Pipeline,
    pipeline_tri_nocull_msaa: Pipeline,
    pipeline_tri_sky_msaa: Pipeline,
    pipeline_tri_wire_msaa: Pipeline,
    pipeline_line_msaa: Pipeline,
    pipeline_point_msaa: Pipeline,
    pipeline_wide_line_msaa: Pipeline,
    /// MSAA sample count for the surface pass: 1 = off, else `MSAA_SAMPLES`.
    msaa: u32,
    /// See [`set_specular_antialiasing`](Self::set_specular_antialiasing).
    specular_antialiasing: bool,
    tone_mapping: ToneMapping,
    /// Whether this pass's destination is the picture itself, and so gets
    /// [`tone_mapping`](Self::tone_mapping). three.js applies tone mapping to
    /// the final output only: a pass into a [`RenderTarget`](super::RenderTarget)
    /// that does not [present](super::RenderTarget::presents) — and into a
    /// cube face — stores untone-mapped light.
    tone_map_pass: bool,
    exposure: f32,
    /// Lazily-created multisampled color + depth targets, keyed by size.
    msaa_targets: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// The previous size's, kept for a frame that alternates two sizes — a
    /// multisampled canvas and a multisampled render target — rather than
    /// reallocating both every frame.
    msaa_targets_spare: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// Whether the last frame's main pass drew into `msaa_targets` — which
    /// attachments [`overlay_attachments`](Self::overlay_attachments) hands out.
    overlay_msaa: bool,
    /// Whether the last frame drawn multisampled. A frame with `Oit` or
    /// `Refract` glass cannot, whatever `msaa` asks; `render_to_multisampled`
    /// then antialiases it with FXAA instead (`fxaa_fallback`).
    last_frame_msaa: bool,
    /// What FXAA reads from in that fallback: a copy of the target.
    fxaa_scratch: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Half-float color-target variants for EffectComposer / postfx RTs.
    pipeline_tri_f16: Pipeline,
    pipeline_tri_alpha_f16: Pipeline,
    pipeline_tri_nocull_f16: Pipeline,
    pipeline_tri_sky_f16: Pipeline,
    pipeline_tri_wire_f16: Pipeline,
    pipeline_line_f16: Pipeline,
    pipeline_point_f16: Pipeline,
    pipeline_point_alpha_f16: Pipeline,
    pipeline_wide_line_f16: Pipeline,
    pipeline_wide_line_alpha_f16: Pipeline,
    pipeline_sprite_f16: Pipeline,
    pipeline_instanced_f16: Pipeline,
    pipeline_skinned_f16: Pipeline,
    pipeline_shadow: Pipeline,
    pipeline_shadow_spot: Pipeline,
    pipeline_shadow_point: Pipeline,
    /// The three shadow passes for skinned casters: the posed mesh, on the
    /// skinned layout.
    pipeline_shadow_skinned: Pipeline,
    pipeline_shadow_spot_skinned: Pipeline,
    pipeline_shadow_point_skinned: Pipeline,
    /// The three shadow passes for an `InstancedMesh` caster: the instance
    /// matrices (and per-instance morphs) from vertex slot 1.
    pipeline_shadow_instanced: Pipeline,
    pipeline_shadow_spot_instanced: Pipeline,
    /// The directional and spot shadow pipelines for cut-out casters.
    shadow_cut: ShadowCutPipelines,
    pipeline_shadow_point_instanced: Pipeline,
    pipeline_instanced: Pipeline,
    pipeline_skinned: Pipeline,
    /// `wireframe: true` variants of the instanced and skinned pipelines:
    /// LineList over the mesh's wire index buffer, like `pipeline_tri_wire`.
    pipeline_instanced_wire: Pipeline,
    pipeline_instanced_wire_f16: Pipeline,
    /// Transparent instanced and skinned draws: alpha-blended, no depth
    /// write, as `pipeline_tri_alpha` is for plain meshes.
    pipeline_instanced_alpha: Pipeline,
    pipeline_instanced_alpha_f16: Pipeline,
    pipeline_skinned_alpha: Pipeline,
    pipeline_skinned_alpha_f16: Pipeline,
    /// The blended draws' pipelines into the `Unorm` view of an sRGB target,
    /// for the encoded-blend pass (see `encoded_pass`): sprite, transparent
    /// instanced, transparent skinned. `None` on a target that is not sRGB.
    encoded_blend_pipelines: Option<[Pipeline; 8]>,
    pipeline_skinned_wire: Pipeline,
    pipeline_skinned_wire_f16: Pipeline,
    /// Post-fx fullscreen-quad pipeline. Reads one input texture, writes to
    /// one output attachment, applies the effect selected by the postfx
    /// uniform's effect-kind field.
    pipeline_postfx: Pipeline,
    /// Additive-blend variant of the postfx pipeline. Used for bloom composite
    /// where the bloom contribution adds to the existing framebuffer content.
    pipeline_postfx_add: Pipeline,
    /// Outline overlay: SrcAlpha + One (three.js AdditiveBlending).
    pipeline_postfx_outline_add: Pipeline,
    /// Multiply-blend variant (DstColor * SrcColor). Used for SSAO composite.
    pipeline_postfx_multiply: Pipeline,
    /// Half-float postfx variants for outline intermediates (Rgba16Float RTs).
    pipeline_postfx_f16: Pipeline,
    pipeline_postfx_add_f16: Pipeline,
    postfx_bgl: wgpu::BindGroupLayout,
    postfx_sampler: wgpu::Sampler,
    postfx_nearest_sampler: wgpu::Sampler,
    /// Repeat + nearest sampler for SSAO noise (matches three.js RepeatWrapping).
    postfx_noise_sampler: wgpu::Sampler,
    /// Pipelines for [`Renderer::run_postfx`], built on first use.
    postfx_ext: postfx_op::PostFxExt,
    /// Uploaded by SSAOPass — 32 hemisphere samples (vec4 stride).
    ssao_kernel_buffer: wgpu::Buffer,
    ssao_noise_tex: wgpu::Texture,
    ssao_noise_view: wgpu::TextureView,
    /// GlitchPass's displacement map (DigitalGlitch `tDisp`): R32Float,
    /// `dt_size` square, read nearest as three.js's `DataTexture` is.
    glitch_disp_tex: wgpu::Texture,
    glitch_disp_view: wgpu::TextureView,
    /// RGBA8 staging for [`Renderer::blit_rgba8`].
    rgba_upload_tex: wgpu::Texture,
    rgba_upload_view: wgpu::TextureView,
    /// Alpha-blended 2D overlay pass, built on the first caption draw so
    /// projects that never draw one allocate nothing.
    /// Only [`Renderer::draw_caption_overlay`] uses this, so without the
    /// `captions` feature there is nothing to hold it for.
    #[cfg(feature = "captions")]
    caption_pass: Option<super::caption_pass::CaptionPass>,
    /// A second instance for [`Renderer::draw_overlay`]. Separate from the
    /// caption one because that pass owns a persistent texture and the overlay
    /// path uploads only the rows that changed — sharing one would mean each
    /// drawing over the other's half-updated contents.
    overlay_pass: Option<super::caption_pass::CaptionPass>,
    mesh_skinned_bgl: wgpu::BindGroupLayout,
    sprite_quad: GpuMesh,
    frame_bgl: wgpu::BindGroupLayout,
    mesh_bgl: wgpu::BindGroupLayout,
    tex_bgl: wgpu::BindGroupLayout,
    env_bgl: wgpu::BindGroupLayout,
    /// three.js's `RectAreaLight` lookup tables (`RectAreaLightUniformsLib`'s
    /// `ltc_1`, `ltc_2`) as the two layers of one texture — see [`upload_ltc`].
    ltc: (wgpu::Texture, wgpu::TextureView),
    /// Group 3 for the refraction pass: the screen-space capture in place of the
    /// shadow maps `fs_ss_glass` never samples, so the pass fits in WebGL2's
    /// sixteen texture units.
    env_bgl_glass: wgpu::BindGroupLayout,
    /// Group 2 for shader materials: the sampler and the displacement slot only,
    /// shared by every custom draw because neither is per-material for a
    /// `ShaderMaterial`.
    custom_tex_bind_group: wgpu::BindGroup,
    /// Its layout, for the per-draw variant a morphed `ShaderMaterial` binds
    /// (the geometry's morph texture at binding 10).
    custom_tex_bgl: wgpu::BindGroupLayout,
    /// The mesh shader and pipeline layout, kept to build the depth-biased
    /// triangle pipelines three.js's `polygonOffset` asks for, on demand.
    mesh_shader: wgpu::ShaderModule,
    mesh_pipeline_layout: wgpu::PipelineLayout,
    /// Those pipelines, by (topology code — see `build_biased_pipeline` —, target
    /// variant: 0 colour format, 1 half float, 2 multisampled; factor bits,
    /// units).
    biased_pipelines: HashMap<(u8, u8, u32, i32), Pipeline>,
    /// The descriptors of every material pipeline, and the variants built
    /// from them for materials with a non-default `depthTest` / `depthWrite`
    /// / `depthFunc` / `colorWrite`. See `pipeline_state`.
    state_pipelines: super::pipeline_state::StatePipelines,
    /// A `Unorm` view of the multisampled colour, when that is sRGB — the
    /// encoded-blend pass draws through it.
    msaa_color_unorm: Option<wgpu::TextureView>,
    /// Where the encoded-blend pass draws on a device that cannot view an
    /// sRGB texture as `Unorm` (WebGL2): a `Unorm` copy of the target, made
    /// and copied back by `encoded_copy`'s two passes. `(width, height, view)`.
    encoded_scratch: Option<(u32, u32, wgpu::TextureView)>,
    /// The scratch copy's pipelines: in (sRGB target → `Unorm` scratch, bytes
    /// kept) and out (back again), with their bind group layout.
    encoded_copy: Option<(wgpu::BindGroupLayout, Pipeline, Pipeline)>,
    /// Bind group layout for custom-shader draws: mesh uniform (binding 0) plus
    /// `ShaderMaterial` user data (binding 1 uniform + bindings 2-4 storage), all
    /// in @group(1) to stay within WebGPU's 4-bind-group limit.
    custom_mesh_bgl: wgpu::BindGroupLayout,
    /// Pipeline layout for custom-shader materials ([frame, custom_mesh, tex, env]).
    /// `None` where the device cannot hold the custom layout; shader materials
    /// are then skipped rather than failing validation on every draw.
    custom_pipeline_layout: Option<wgpu::PipelineLayout>,
    /// The same for a screen-space shader material
    /// (`ShaderMaterial::screen_space`): its group 3 is the refraction pass's
    /// (`env_bgl_glass`), which carries the capture at bindings 9–12 that the
    /// main layout leaves out.
    custom_ss_pipeline_layout: Option<wgpu::PipelineLayout>,
    /// Where the environment is trimmed (WebGL2), the layout of a custom shader
    /// that reads shadows or area lights, and its group 1. The sixteen texture
    /// units hold the full environment or the volume slots but not both, so
    /// this one has the first and `custom_pipeline_layout` the second.
    custom_lit_pipeline_layout: Option<wgpu::PipelineLayout>,
    custom_lit_mesh_bgl: Option<wgpu::BindGroupLayout>,
    /// By shader hash, whether it draws through `custom_lit_pipeline_layout`.
    custom_lit: HashMap<u64, bool>,
    /// Lazily-compiled custom-shader pipelines. See [`CustomPipelineKey`].
    custom_pipelines: HashMap<CustomPipelineKey, Pipeline>,
    /// Transparent custom-shader pipelines into the `Unorm` view of an sRGB
    /// target, for the encoded-blend pass; keyed as `custom_pipelines` is,
    /// less the framebuffer flag.
    custom_encoded_pipelines: HashMap<(u64, bool, bool, u32), Pipeline>,
    /// Custom pipelines multisampled, for an opaque shader material in a
    /// frame drawn under MSAA; keyed as `custom_encoded_pipelines` is.
    custom_msaa_pipelines: HashMap<(u64, bool, bool, u32), Pipeline>,
    /// Shaders that would not compile. Remembered so the error is logged once
    /// rather than every frame, and so the draw is skipped rather than retried.
    custom_failed: std::collections::HashSet<CustomPipelineKey>,
    depth_view: wgpu::TextureView,
    /// The renderer's own depth texture (COPY_SRC) for the direct-`render` path.
    depth_texture: wgpu::Texture,
    /// 1×1 placeholder depth texture for postfx binding 6 when hw depth is unused.
    hw_depth_tex: wgpu::Texture,
    depth_size: (u32, u32),

    geom_cache: IdentityKeyed<*const BufferGeometry, CachedGpuMesh>,
    /// Whether to skip colour-pass draws the camera cannot see. On by default,
    /// matching three.js, where `Object3D.frustumCulled` starts true.
    frustum_culling: bool,
    /// `(drawn, culled)` for the last frame. Culling is defined to leave the
    /// image alone, so a pixel comparison cannot tell whether it did anything
    /// — this is the only way to see that it is working, and the only way a
    /// test can tell a correct frustum from one that never rejects anything.
    cull_stats: (usize, usize),
    /// GPU time for the main pass, when the adapter can measure it.
    timer: Option<GpuTimer>,
    /// GPU time from the first shadow pass to the last, when one ran.
    shadow_timer: Option<GpuTimer>,
    /// three.js `shadowMap.autoUpdate` / `needsUpdate`.
    shadow_map: super::viewport::ShadowMap,
    /// What the shadow maps were last drawn for — which caster kinds, at
    /// which directional map size. A render needing anything else redraws
    /// them whatever [`shadow_map`](Self::shadow_map) says.
    shadow_sig: Option<ShadowSig>,
    /// What the shadow maps were last drawn from (`shadow_fingerprint`), or
    /// `None` when it could not be told — skinned, instanced or custom-shader
    /// casters, which change without anything here changing.
    /// Each shadow pass's content hash when last drawn (see the update
    /// below), so an unchanged pass is skipped.
    shadow_pass_hashes: Vec<Option<u64>>,
    /// Shadow passes drawn and caster draws issued into them, ever.
    shadow_stats: (u64, u64),
    /// Renders that drew the shadow maps.
    shadow_updates: u64,
    skin_attr_cache: IdentityKeyed<*const BufferGeometry, wgpu::Buffer>,
    /// Morph-target textures, by geometry (rebuilt on a version change).
    morph_cache: IdentityKeyed<*const BufferGeometry, MorphTexture>,
    /// Each `InstancedMesh`'s instance buffers (matrices and tints in slot
    /// 0, custom columns in slot 1), kept across frames and rewritten only
    /// when the data changes — they used to be allocated anew every frame.
    instance_cache: HashMap<(crate::core::ObjectId, u8), InstanceBuffer>,
    /// Counts `render_inner` calls; an instance buffer unused for a few of
    /// them belonged to a mesh no longer drawn and is dropped.
    instance_frame: u64,
    /// Source of `InstanceBuffer::stamp`.
    instance_stamps: u64,
    /// Each geometry's [`UvSetTexture`], for those with `uv1`/`uv2`/`uv3`.
    uv_cache: IdentityKeyed<*const BufferGeometry, UvSetTexture>,
    /// Bound for geometry without extra uv sets (they read zero, as an
    /// attribute a geometry lacks does in three.js).
    default_uv_sets: (wgpu::Texture, wgpu::TextureView),
    /// Bound at the morph slot for every draw without morph targets.
    default_morph: (wgpu::Texture, wgpu::TextureView),
    /// Wireframe line lists for partial spans (a group or a draw range of a
    /// `wireframe: true` mesh), by (geometry, version, start, end).
    wire_span_cache: IdentityKeyed<(*const BufferGeometry, u32, u32, u32), (wgpu::Buffer, u32)>,
    /// Uploaded textures, by [`Texture::id`].
    tex_cache: HashMap<TexCacheKey, GpuTexture>,
    /// The `instance_frame` each cached texture was last asked for. A texture
    /// nothing has drawn for [`cache_retention`](Self::set_cache_retention)
    /// frames was dropped by its owner, or will be uploaded again if not:
    /// without this, every texture ever drawn kept its GPU copy for good.
    tex_last_used: HashMap<TexCacheKey, u64>,
    /// Frames an unused texture keeps its GPU copy; `u64::MAX` keeps it.
    cache_retention: u64,
    /// Textures [`preload_texture`](Self::preload_texture) put on the GPU, by
    /// id: never evicted, since their handles have no pixels to restore them.
    tex_pinned: std::collections::HashSet<u64>,
    /// For a streamed texture, by [`Texture::id`]: the frame count of the last
    /// frame written into its GPU texture.
    stream_uploads: HashMap<u64, u64>,
    /// Rows turned over for a streamed frame with `flip_y`, kept between frames.
    stream_scratch: Vec<u8>,
    /// Converts streamed YUV frames (video) into their textures on the GPU.
    yuv: Option<super::yuv::YuvConverter>,
    /// WebGL2 only: the canvas a browser `VideoFrame` is drawn into to be
    /// copied, kept between frames.
    #[cfg(target_arch = "wasm32")]
    frame_canvas: Option<(
        web_sys::HtmlCanvasElement,
        web_sys::CanvasRenderingContext2d,
    )>,
    /// Builds mip chains with the GPU rather than the CPU. See [`MipGenerator`].
    mipgen: Option<MipGenerator>,
    /// Lookup table for render-target-backed textures, keyed by RT id.
    rt_view_cache: HashMap<u32, wgpu::TextureView>,
    /// Linear (non-sRGB-decode) sample views for AO/depth/normal postfx buffers.
    rt_linear_view_cache: HashMap<u32, wgpu::TextureView>,
    /// Render targets keyed by id (for creating per-pass sample views).
    rt_registry: HashMap<u32, Arc<super::RenderTarget>>,
    /// GPU depth-buffer views for SSAO (Depth32Float, depth-only aspect).
    rt_depth_view_cache: HashMap<u32, wgpu::TextureView>,
    /// Lookup table for cube render targets' cube views, keyed by RT id.
    cube_rt_view_cache: HashMap<u32, wgpu::TextureView>,
    /// Highest mip index of the cube currently bound to `env_tex`.
    ///
    /// Carried in a uniform because the shader cannot ask: `textureNumLevels`
    /// has no GLSL ES 3.0 equivalent, so on WebGL2 the chain length has to come
    /// from the CPU. Recomputed when the environment bind group is rebuilt and
    /// applied every frame, since `FrameUniforms` starts from `Default`.
    env_max_mip: f32,
    cube_cache: HashMap<super::gpu_texture::CubeCacheKey, GpuCubeTexture>,
    env_bind_group: wgpu::BindGroup,
    shadow_env_bind_group: wgpu::BindGroup,
    /// The custom pipeline's trimmed group-3 layout and the bind group that
    /// fills it, both `Some` only where the environment is trimmed for the
    /// volume slots (WebGL2). Elsewhere a custom draw binds `env_bind_group`.
    /// The layout is kept to rebuild the bind group when the scene environment
    /// changes; the uniform (binding 13) is the shared `env_uniform`.
    custom_env_bgl: Option<wgpu::BindGroupLayout>,
    custom_env_bind_group: Option<wgpu::BindGroup>,
    /// The scene environment's `EnvUniforms`, written each frame.
    env_uniform: wgpu::Buffer,
    /// Group 3 for Standard/Physical materials with an `envMap` of their own,
    /// keyed by the cube, its intensity and its rotation. Rebuilt with the
    /// scene's group, whose shadow-map views they share.
    material_env_groups: HashMap<MaterialEnvKey, (wgpu::Buffer, wgpu::BindGroup)>,
    env_sampler: wgpu::Sampler,
    default_env_cube: GpuCubeTexture,
    default_cube_uv_view: wgpu::TextureView,
    /// The shadow passes' uniforms and group 0 — see
    /// `ensure_shadow_pass_buffer` — and the layout and slot stride.
    shadow_pass_buffer: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// The opaque depth redrawn single-sampled (see `depth_replay` where it
    /// is built), and its pass uniform: the camera's view-projection.
    depth_replay: Vec<Pipeline>,
    depth_replay_pass: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Redraw the depth even where it could be resolved: so a test on a
    /// native device runs the path WebGL2 takes.
    force_depth_replay: bool,
    /// This render is a multisampled render target's (`render_to` with
    /// `RenderTarget::samples`): the target's depth, which the multisampled
    /// depth is resolved into and the glass march copies.
    msaa_depth_out: Option<wgpu::Texture>,
    shadow_pass_bgl: wgpu::BindGroupLayout,
    shadow_pass_stride: u64,
    /// three.js `renderer.shadowMap.type`; a light's
    /// `ShadowSettings::filter` overrides it.
    shadow_type: crate::lights::ShadowType,
    env_cached_key: Option<super::gpu_texture::CubeCacheKey>,
    /// Rebuild the environment bind group on the next frame regardless of the
    /// cache keys.
    ///
    /// Needed because "invalidate" cannot be expressed by clearing those keys: a
    /// scene with no environment map already has `None` for both, so `None !=
    /// None` detects nothing and the group survives a shadow-texture
    /// reallocation still pointing at the old view.
    env_dirty: bool,
    /// When the env source is a CubeRenderTarget, this is its id. Used to
    /// invalidate the env bind group when the RT changes.
    env_cached_rt_id: Option<u32>,
    /// The shadow maps: a layer per shadow-casting light of each kind, six
    /// per point light. See [`ShadowArray`].
    dir_shadows: ShadowArray,
    spot_shadows: ShadowArray,
    point_shadows: ShadowArray,
    /// True while rendering one face of an environment cube. Screen-space
    /// passes that assume the frame's dimensions are skipped.
    cube_pass: bool,
    /// `set_viewport` / `set_scissor` / `set_scissor_test` / pixel ratio /
    /// auto-clear. See [`super::viewport`].
    viewport_state: super::viewport::ViewportState,
    shadow_sampler: wgpu::Sampler,
    spot_shadow_sampler: wgpu::Sampler,
    point_shadow_sampler: wgpu::Sampler,

    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    /// The frame group with the stand-in capture, for the capture passes.
    frame_bind_group_dummy: wgpu::BindGroup,
    frame_buffer_for_bg: wgpu::Buffer,
    transmission_sampler: wgpu::Sampler,
    transmission: Option<TransmissionCapture>,
    /// Which capture pass is running: `None` outside them, `Some(false)` the
    /// opaque scene, `Some(true)` double-sided transmissive back faces.
    capturing_transmission: Option<bool>,
    /// See [`set_reflective_transparency`](Self::set_reflective_transparency).
    reflective_transparency: bool,
    /// Per draw: its uniforms' offset in `mesh_arena`, its mesh bind group
    /// (over that slice) and its texture bind group.
    per_mesh: Vec<(u64, wgpu::BindGroup, std::sync::Arc<wgpu::BindGroup>)>,
    /// Every draw's `MeshUniforms`, one after another at `mesh_stride`, written
    /// with one call a frame. One small buffer per draw meant one
    /// `write_buffer` per draw — two thousand meshes, two thousand staging
    /// copies, and the largest single cost of a frame on the CPU.
    mesh_arena: Option<wgpu::Buffer>,
    /// `MeshUniforms` rounded up to the device's uniform offset alignment.
    mesh_stride: u64,
    /// The frame's uniforms on their way to `mesh_arena`.
    mesh_bytes: Vec<u8>,
    /// Texture bind groups, keyed on the textures they bind.
    ///
    /// These used to be rebuilt from scratch once per draw per frame — ten
    /// texture views and two samplers looked up and a `create_bind_group` for
    /// every mesh, every frame, whether or not anything about it had changed.
    /// Measured at 0.3 ms for 73 draws, and it grows with the draw count, so
    /// a scene with several hundred meshes was paying milliseconds a frame to
    /// rebuild bind groups that were bit-for-bit what they had been.
    ///
    /// A bind group is a pure function of the textures it binds, so this is
    /// keyed on exactly those and invalidated whenever a texture is uploaded,
    /// evicted, or re-registered as a render target.
    tex_bg_cache: HashMap<TexBgKey, std::sync::Arc<wgpu::BindGroup>>,
    /// The main shader module, on the legacy material-map layout — the module
    /// a material shader variant replaces.
    main_shader: wgpu::ShaderModule,
    /// Material shader variants built so far: each one's module and group-2
    /// layout (see `material_maps`).
    map_variants:
        HashMap<super::material_maps::MapVariant, (wgpu::ShaderModule, wgpu::BindGroupLayout)>,
    /// three.js `WebGLCubeUVMaps`: each cube texture lighting a
    /// Standard/Physical material, prefiltered through PMREM once, by its
    /// source's upload key.
    pmrem_cache: HashMap<super::gpu_texture::CubeCacheKey, std::sync::Arc<CubeTexture>>,
    /// Group-2 bind groups of variant draws.
    variant_bg_cache: HashMap<(VariantBgKey, (usize, u32)), std::sync::Arc<wgpu::BindGroup>>,

    sampler_linear_repeat: wgpu::Sampler,
    /// Material map samplers, by [`sampler_index`].
    samplers: Vec<wgpu::Sampler>,

    // Default fallbacks (1x1 textures used when a material slot is empty).
    default_white_srgb: GpuTexture,
    default_white_linear: GpuTexture,
    /// 1x1 `Rgba32Float`, bound to the data-texture slots on devices that have
    /// storage buffers and therefore never read them. The layout declares the
    /// slots either way, so every backend gets the same bind group shape.
    default_data_tex_view: wgpu::TextureView,
    /// 1x1x1 `R8Unorm` 3D texture for a `ShaderMaterial`'s empty volume slots.
    default_volume_view: wgpu::TextureView,
    /// Linear, clamp-to-edge: a volume's `u_vol_samp`. Repeat would wrap the
    /// far face of a scan onto the near one at the boundary.
    sampler_volume: wgpu::Sampler,
    /// `(near, far)` of the camera the last frame was drawn with.
    last_camera_planes: (f32, f32),
    /// `(tan(fov/2), aspect)` of the same camera, read back off its projection.
    last_camera_lens: (f32, f32),
    default_normal: GpuTexture, // (0.5, 0.5, 1.0, 1.0) — straight up tangent normal
    /// The empty cloud-shadow slot. An opaque black default would not do: a cover map
    /// is read from *alpha* — so an unset slot there would read as total
    /// overcast and black out the planet.
    default_transparent: GpuTexture,

    // --- Screen-space refraction glass (TransparencyMode::Refract) ---
    /// Mipmapped capture of the opaque scene, sampled by the glass shader for
    /// screen-space refraction/reflection. `(w, h, texture, all-mips view,
    /// per-mip render views, mip count, format)`.
    ss_color: Option<SsColor>,
    /// Filtering sampler (trilinear) for `ss_color` — roughness selects the LOD.
    ss_glass_sampler: wgpu::Sampler,
    /// Sampleable copy of the opaque depth (COPY_DST | TEXTURE_BINDING), marched
    /// by the glass shader. `(w, h, texture, view)`.
    ss_depth: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    /// Nearest glass **back-face** depth (RENDER_ATTACHMENT | TEXTURE_BINDING),
    /// used to derive the per-pixel glass thickness for two-surface refraction.
    ss_back_depth: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    /// Scratch the size and format of the capture, which a layer of
    /// overlapping glass draws into before it becomes the next layer's
    /// capture (see `set_refraction_layers`).
    ss_layer: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    /// How many layers overlapping refraction glass composites in.
    refraction_layers: usize,
    /// Depth-only pipeline rendering glass back faces (cull front) into ss_back_depth.
    pipeline_ss_back_depth: Pipeline,
    /// Fullscreen "sample one texture → write" pipeline (mip downsample + blit),
    /// one per framebuffer format.
    pipeline_ss_blit: Pipeline,
    pipeline_ss_blit_f16: Pipeline,
    /// Glass surface pipelines (screen-space refraction/reflection).
    pipeline_ss_glass: Pipeline,
    pipeline_ss_glass_f16: Pipeline,
    /// `fs_glass_shadow`: each light's shadow on the refracting surfaces,
    /// into `ss_glass_shadow`.
    pipeline_glass_shadow: Pipeline,
    ss_glass_shadow: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// BGL for the blit/downsample pipeline (texture + sampler).
    ss_blit_bgl: wgpu::BindGroupLayout,

    // --- Temporal anti-aliasing (opt-in via `set_taa`) ---
    taa_enabled: bool,
    taa_frame: u32,
    /// Accumulate a running mean instead of a feedback blend — see the converge
    /// branch in `TAA_SHADER`, and [`Renderer::set_taa_converge`].
    taa_converge: bool,
    /// Whether the accumulated image is copied back into the frame target this
    /// render. Lowered for every sample but the last of an accumulation run —
    /// see [`Renderer::set_taa_blit`].
    taa_blit: bool,
    /// Format the history buffers were built with; see `ensure_taa_targets`.
    taa_hist_format: Option<wgpu::TextureFormat>,
    /// Sub-pixel offset to use instead of the Halton sequence, in pixels.
    ///
    /// For a caller doing its own accumulation — it needs the jitter without
    /// the temporal pass that normally comes with it.
    jitter_override: Option<(f32, f32)>,
    /// Views of the colour and depth the temporal pass reads, and its two bind
    /// groups — one per history parity — kept across samples.
    ///
    /// An accumulation run calls `render` `n` times against textures that do
    /// not change, and these were being rebuilt every one of them: two texture
    /// views and two bind groups a sample, sixteen samples a frame, three
    /// hundred frames. None of it depends on anything but the targets.
    taa_bind_cache: Option<TaaBindCache>,
    taa_prev_view_proj: [f32; 16],
    /// Camera matrices tracked EVERY frame, not just when TAA is on, so a pass
    /// running after the render can reproject: `mb_inv_vp` inverts the frame
    /// just drawn and `mb_prev_vp` is the one before it. Motion blur needs both
    /// and must not depend on whether temporal AA happens to be enabled.
    mb_inv_vp: [f32; 16],
    mb_prev_vp: [f32; 16],
    mb_last_vp: [f32; 16],
    /// Camera position at the previous TAA frame, for the sky reprojection.
    taa_prev_cam: [f32; 3],
    /// Ping-pong history: `(w, h, texA, viewA, texB, viewB)`; frame parity picks
    /// read vs. write.
    taa_history: Option<(
        u32,
        u32,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    )>,
    taa_uniform: wgpu::Buffer,
    taa_bgl: wgpu::BindGroupLayout,
    taa_sampler: wgpu::Sampler,
    pipeline_taa: Pipeline,
    pipeline_taa_f16: Pipeline,

    color_format: wgpu::TextureFormat,
}

/// Mipmapped opaque-scene capture used by the screen-space glass pass.
struct SsColor {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    /// View over all mips — bound to the glass shader for LOD sampling.
    all_mips_view: wgpu::TextureView,
    /// One single-mip view per level — render targets for the downsample chain.
    mip_render_views: Vec<wgpu::TextureView>,
    /// One single-mip sample view per level — sources for the downsample chain.
    mip_sample_views: Vec<wgpu::TextureView>,
    /// Mip 0 as its `Unorm` twin when the capture is sRGB: the encoded-blend
    /// pass draws a refraction frame's blended surfaces into it.
    mip0_unorm: Option<wgpu::TextureView>,
    mips: u32,
    format: wgpu::TextureFormat,
}

/// A hemisphere of sample offsets for SSAO, clustered towards the origin.
///
/// Points are drawn on the +Z hemisphere — the shader rotates them into the
/// fragment's tangent frame — and scaled by a quadratic so that most samples
/// sit close in, which is where occlusion actually varies.
fn default_ssao_kernel() -> Vec<f32> {
    let mut out = Vec::with_capacity(32 * 3);
    // A fixed integer hash rather than a PRNG crate: the kernel has to be the
    // same on every run or the same scene renders differently each launch.
    let hash = |i: u32, salt: u32| -> f32 {
        let mut h = i
            .wrapping_mul(747_796_405)
            .wrapping_add(salt.wrapping_mul(2_891_336_453));
        h ^= h >> 16;
        h = h.wrapping_mul(2_246_822_519);
        h ^= h >> 13;
        (h >> 8) as f32 / 16_777_216.0
    };
    for i in 0..32u32 {
        let mut v = [
            hash(i, 1) * 2.0 - 1.0,
            hash(i, 2) * 2.0 - 1.0,
            // Hemisphere: never below the surface.
            hash(i, 3),
        ];
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-4);
        // Scale in towards the origin, quadratically.
        let t = i as f32 / 32.0;
        let scale = (0.1 + 0.9 * t * t) / len;
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
        out.extend_from_slice(&v);
    }
    out
}

/// Sixteen rotation values, one per texel of the 4x4 noise texture.
///
/// Never zero: the shader builds its tangent from this, and a zero would make
/// it `normalize(vec3(0))` — a NaN that spreads through the whole kernel.
fn default_ssao_noise() -> Vec<f32> {
    (0..16)
        .map(|i| {
            let mut h = (i as u32).wrapping_mul(2_654_435_761);
            h ^= h >> 15;
            h = h.wrapping_mul(2_246_822_519);
            h ^= h >> 13;
            let u = (h >> 8) as f32 / 16_777_216.0;
            // Away from zero, and signed, so neighbouring texels decorrelate.
            if u < 0.5 {
                -0.35 - u
            } else {
                0.35 + (u - 0.5)
            }
        })
        .collect()
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One depth texture's values written as another's, by a fullscreen triangle:
/// where depth cannot be copied (WebGL2). Sampled, not loaded — `textureLoad`
/// of a depth texture reads 0 there — and bound as an unfilterable float
/// texture, not a depth one: naga writes `textureSampleLevel` of a
/// `texture_depth_2d` as GLSL `textureLod` with an integer level, which
/// GLSL ES rejects, and the pipeline never compiles.
const DEPTH_COPY_SHADER: &str = crate::wgsl!(
    r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
    let uv = p.xy / vec2<f32>(textureDimensions(src));
    return textureSampleLevel(src, samp, uv, 0.0).r;
}
"#
);

/// Hardware MSAA sample count used when antialiasing is enabled for the
/// direct-to-surface opaque forward pass. 4× is universally supported.
const MSAA_SAMPLES: u32 = 4;

/// One step of a cheap word mix (FxHash's): for fingerprints over trusted
/// data, run every frame, where SipHash's resistance to chosen input buys
/// nothing.
fn mix_word(h: u64, x: u64) -> u64 {
    (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95)
}

/// A sequence of words, mixed.
fn hash_words(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325, mix_word)
}
/// The spot shadow map's size (and the directional map's until a light asks
/// for another), texels.
const SPOT_SHADOW_SIZE: u32 = 4096;
/// Each face of the point shadow cube, texels.
const POINT_SHADOW_SIZE: u32 = 512;
/// The OIT revealage: the layers' transmittance multiplied up per channel,
/// and their coverage's complement in alpha.
const OIT_REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// `make_glass_pipeline`'s passes: the depth prepass, and the tint and light
/// halves of the colour (`GLASS_OUTPUT` 1 and 2 in the shader).
const GLASS_DEPTH: u8 = 0;
const GLASS_TINT: u8 = 1;
const GLASS_LIGHT: u8 = 2;
/// What `fs_glass_shadow` packs each light's shadow on the glass into.
const GLASS_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

/// What the shadow maps were last drawn for: the casters of each kind and
/// the directional and spot map sizes.
type ShadowSig = (usize, usize, usize, u32, u32);

/// One kind's shadow maps: a 2D depth array, a layer per light (six per
/// point light, its cube faces), sampled through one view.
struct ShadowArray {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    layers: u32,
}

impl ShadowArray {
    fn new(device: &wgpu::Device, label: &str, size: u32, layers_needed: u32) -> Self {
        // Never one layer, nor a multiple of six on a square texture: the GL
        // backend makes those a plain 2D texture or a cube map, and neither
        // can be bound as the 2D array the shader declares.
        let mut layers = layers_needed.max(2);
        if layers.is_multiple_of(6) {
            layers += 1;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            size,
            layers,
        }
    }

    /// Layer `layer`, as a pass's depth attachment.
    fn layer_view(&self, layer: u32) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers shadow layer"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShadowKind {
    Dir,
    Spot,
    Point,
}

/// A shadow-casting light's shadow, as the shader's `ShadowRecord`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShadowRecordGpu {
    /// `[kind (0 directional, 1 spot, 2 point), first layer, cascades, filter]`.
    info: [u32; 4],
    /// `[depth bias, normal bias, filter radius, one texel]`, the last two in
    /// map uv (direction units for a point light).
    params: [f32; 4],
    /// Directional: each cascade's far edge, view depth. Point: position and
    /// radius.
    extra: [f32; 4],
    /// PCSS: `[search radius, spot near, spot far, penumbra scale]`.
    pcss: [f32; 4],
    /// World → light clip: a cascade each, or the light's.
    vp: [[f32; 16]; 4],
}

/// What one shadow pass draws with, at its own offset in one buffer — the
/// shader's `ShadowPass`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShadowPassGpu {
    vp: [f32; 16],
    light: [f32; 4],
}

/// One shadow pass: a light's layer and the matrix it is drawn with (for a
/// point light, one cube face, with the light's position and radius).
struct ShadowPass {
    kind: ShadowKind,
    layer: u32,
    vp: [f32; 16],
    pos: [f32; 4],
}

/// Cube face `face` of a point light's shadow: world → clip.
fn point_face_vp(pos: Vector3, radius: f32, face: usize) -> [f32; 16] {
    let face_lookat: [(Vector3, Vector3); 6] = [
        (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
        (Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
        (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)),
        (Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)),
        (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, -1.0, 0.0)),
        (Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, -1.0, 0.0)),
    ];
    let fov = std::f32::consts::FRAC_PI_2;
    let proj = Matrix4::perspective(fov, 1.0, 0.1, radius.max(1.0));
    let (forward, up) = face_lookat[face];
    let view = Matrix4::look_at(pos, pos + forward, up);
    // NOTE: the Metal backend negates clip-space x here and this
    // path must not — the two disagree about the cube's handedness
    // somewhere further down, and mirroring this one takes its
    // shadow from weak to absent (0.0021 of darkening to 0.0000).
    // That asymmetry is unexplained and is the thread to pull on
    // next; see `metal_casts_a_point_shadow`.
    // Negate clip-space y.
    //
    // Cube maps are left-handed in every graphics API and
    // `Matrix4::look_at` is right-handed, so a face rendered with
    // it is laid down flipped against what the sampler will later
    // expect. Each face looks perfectly reasonable alone, which is
    // what makes this invisible: the shadow simply lands somewhere
    // else.
    //
    // *Which* axis differs by backend, and that is not a guess —
    // `tests/shadow_fixtures.rs` measures the shadow's distance
    // from where the geometry puts it, in pixels:
    //
    // ```text
    //            none      x       y      xy
    //   wgpu     79.2   133.1     2.1   107.0
    //   metal   107.0     2.1   133.1    79.2
    // ```
    //
    // The two tables are each other's, shifted by one flip: the
    // backends' framebuffer-y conventions are opposite, so the
    // correction is too. Anything but the 2.1 here is a shadow of
    // something that is not there.
    let mut vp = proj.multiply(&view).elements;
    vp[1] = -vp[1];
    vp[5] = -vp[5];
    vp[9] = -vp[9];
    vp[13] = -vp[13];
    vp
}

// ---- ShaderMaterial (custom-shader) support ----------------------------------

/// Cache key for a custom shader = a hash of *both* stages. Keying on the
/// fragment alone would hand two materials that differ only in their vertex
/// source the same compiled pipeline, and the second one would silently draw
/// with the first one's geometry.
/// Mixed into a custom shader's hash when it draws an `InstancedMesh`, whose
/// pipeline has the instanced vertex layout.
const INSTANCED_CUSTOM_SALT: u64 = 0x9e37_79b9_7f4a_7c15;
/// Salted into a custom pipeline's key when its material takes
/// alpha-to-coverage: the same source, another pipeline.
const COVERAGE_CUSTOM_SALT: u64 = 0xc2b2_ae3d_27d4_eb4f;

fn custom_shader_hash(fragment: &str, vertex: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    fragment.hash(&mut h);
    vertex.hash(&mut h);
    h.finish()
}

/// Assemble a full WGSL module for a custom shader: the built-in preamble
/// (everything before the built-in `fs_main` — structs, `vs_main`, bindings for
/// groups 0-3, and all helper functions), then the `@group(4)` user bindings,
/// then the user's fragment.
pub(crate) fn custom_shader_module_source(
    fragment: &str,
    vertex: Option<&str>,
    storage_buffers: bool,
    coverage: bool,
    samples: u32,
) -> String {
    let source = super::shader::legacy_shader_source();
    let preamble = source.split("@fragment").next().unwrap_or(source);
    // User data lives in @group(1) alongside the mesh uniform (binding 0), to
    // stay within WebGPU's 4-bind-group limit.
    let user_bindings = "\n\
struct ThreersUserData { data: array<vec4<f32>, 16>, };\n\
@group(1) @binding(1) var<uniform> u_data: ThreersUserData;\n\
@group(1) @binding(5) var u_tex0: texture_2d<f32>;\n\
@group(1) @binding(6) var u_tex1: texture_2d<f32>;\n\
@group(1) @binding(7) var u_tex2: texture_2d<f32>;\n\
@group(1) @binding(8) var u_tex3: texture_2d<f32>;\n\
@group(1) @binding(9) var u_samp: sampler;\n\
@group(1) @binding(10) var u_s0_tex: texture_2d<f32>;\n\
@group(1) @binding(11) var u_s1_tex: texture_2d<f32>;\n\
@group(1) @binding(12) var u_s2_tex: texture_2d<f32>;\n";
    // The volume slots. Two 3D textures and a sampler — not storage buffers, so
    // WebGL2 has them too; the custom pipeline frees the texture units to hold
    // them there by trimming its environment group (see the custom env layout
    // in `Renderer::new`). They are declared unconditionally because the module
    // is only ever compiled for a device that has the custom pipeline, and that
    // is exactly a device whose layout carries these three (both gate on
    // `caps::supports_shader_material_volumes`). Module, layout and bind group
    // agree on it.
    let volume_bindings = "@group(1) @binding(13) var u_vol0: texture_3d<f32>;\n\
@group(1) @binding(14) var u_vol1: texture_3d<f32>;\n\
@group(1) @binding(15) var u_vol_samp: sampler;\n";
    // WebGL2 has no storage buffers at all, so on that backend these three are
    // left out of the module *and* out of the layout and bind group that go with
    // it — all three have to agree or the pipeline is rejected. A shader that
    // reads `u_s0` simply will not compile there, which is the honest outcome:
    // the data has nowhere to live.
    let storage_bindings = if storage_buffers {
        "@group(1) @binding(2) var<storage, read> u_s0: array<f32>;\n\
@group(1) @binding(3) var<storage, read> u_s1: array<f32>;\n\
@group(1) @binding(4) var<storage, read> u_s2: array<f32>;\n"
    } else {
        ""
    };
    // `s0`/`s1`/`s2` are the portable way to read the storage arrays: they are
    // defined on every backend, and on one without storage buffers they read
    // the same data out of a texture instead. `u_s0`..`u_s2` stay for the
    // shaders that already index them directly, but they only exist where the
    // device does.
    let accessors = if storage_buffers {
        "fn s0(i: u32) -> f32 { return u_s0[i]; }\n\
fn s1(i: u32) -> f32 { return u_s1[i]; }\n\
fn s2(i: u32) -> f32 { return u_s2[i]; }\n\
fn s_len(slot: u32) -> u32 {\n\
    if (slot == 0u) { return arrayLength(&u_s0); }\n\
    if (slot == 1u) { return arrayLength(&u_s1); }\n\
    return arrayLength(&u_s2);\n\
}\n"
    } else {
        "fn s_fetch(t: texture_2d<f32>, i: u32) -> f32 {\n\
    let w = u32(textureDimensions(t).x);\n\
    let texel = i / 4u;\n\
    let p = vec2<i32>(i32(texel % w), i32(texel / w));\n\
    let v = textureLoad(t, p, 0);\n\
    let c = i % 4u;\n\
    if (c == 0u) { return v.x; }\n\
    if (c == 1u) { return v.y; }\n\
    if (c == 2u) { return v.z; }\n\
    return v.w;\n\
}\n\
fn s0(i: u32) -> f32 { return s_fetch(u_s0_tex, i); }\n\
fn s1(i: u32) -> f32 { return s_fetch(u_s1_tex, i); }\n\
fn s2(i: u32) -> f32 { return s_fetch(u_s2_tex, i); }\n\
fn s_len(slot: u32) -> u32 {\n\
    var d = textureDimensions(u_s0_tex);\n\
    if (slot == 1u) { d = textureDimensions(u_s1_tex); }\n\
    if (slot == 2u) { d = textureDimensions(u_s2_tex); }\n\
    return u32(d.x) * u32(d.y) * 4u;\n\
}\n"
    };
    // The pipeline the module is built for, for a shader that draws its own
    // anti-aliasing:
    //
    // - three.js `alphaToCoverage` — a fragment's alpha as the share of the
    //   pixel's samples it covers. `coverage_alpha` is what a shader returns
    //   as its alpha for that: the coverage itself in a multisampled pipeline
    //   with alpha-to-coverage on, a cut at half (discard below, 1 above) in
    //   any other, so the same shader is right in both.
    // - `THREERS_SAMPLES` and `threers_sample_offset(i)`: the samples a pixel
    //   has (1, or 4 under MSAA) and where sample `i` sits from its centre,
    //   in pixels (the standard 4× pattern), for a shader that works out which
    //   samples it covers itself — and returns them in `frag_out(colour,
    //   mask)`, as `@builtin(sample_mask)` where the backend has it
    //   (`THREERS_SAMPLE_MASK`; not WebGL2, where the alpha is the share of
    //   samples, for alpha-to-coverage to place).
    //
    // The same number of lines in every variant, so error lines stay put.
    let mask = samples > 1 && storage_buffers && sample_mask_is_cheap();
    let (mask_field, mask_set) = if mask {
        ("@builtin(sample_mask) mask : u32, ", "o.mask = mask; ")
    } else {
        ("", "")
    };
    let coverage = format!(
        "const THREERS_COVERAGE : bool = {coverage};\n\
const THREERS_SAMPLES : u32 = {samples}u;\n\
const THREERS_SAMPLE_MASK : bool = {mask};\n\
fn coverage_alpha(coverage : f32) -> f32 {{\n\
    if (THREERS_COVERAGE) {{ return clamp(coverage, 0.0, 1.0); }}\n\
    if (coverage < 0.5) {{ discard; }}\n\
    return 1.0;\n\
}}\n\
struct ThreersFragOut {{ @location(0) color : vec4<f32>, {mask_field}}};\n\
fn frag_out(color : vec4<f32>, mask : u32) -> ThreersFragOut {{ var o : ThreersFragOut; o.color = color; {mask_set}return o; }}\n\
fn threers_sample_offset(i : u32) -> vec2<f32> {{\n\
    if (THREERS_SAMPLES == 1u) {{ return vec2<f32>(0.0); }}\n\
    return select(select(select(vec2<f32>(0.125, 0.375), vec2<f32>(-0.375, 0.125), i == 2u), vec2<f32>(0.375, -0.125), i == 1u), vec2<f32>(-0.125, -0.375), i == 0u);\n\
}}\n"
    );
    let vertex = vertex.map(|v| format!("\n{v}\n")).unwrap_or_default();
    format!("{preamble}{user_bindings}{volume_bindings}{storage_bindings}{accessors}{coverage}{vertex}\n{fragment}\n")
}

/// Parse and validate an assembled custom-shader module, before wgpu sees it.
///
/// Worth doing because of *where* the failure otherwise lands. A bad
/// `ShaderMaterial` reaches `create_shader_module`, which does not return an
/// error — it goes to the device's uncaptured-error handler and takes the
/// process down, from inside the draw loop, with line numbers into a module the
/// caller has never seen: the user wrote forty lines and the error says
/// `wgsl:1707`. Checking here turns that into a message naming the user's own
/// line, and lets the renderer skip one material instead of ending the program.
///
/// Set `THREERS_DUMP_SHADER` to a path to also write the assembled module out;
/// that is the only way to read line 1707 for yourself.
/// On `wasm32` there is nothing to check against: naga is not in wgpu's browser
/// build, so the browser's own WGSL compiler does this job and reports to the
/// console. This returns `Ok` there rather than pretending otherwise — see
/// `ShaderMaterial::VALIDATION_AVAILABLE`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn validate_custom_shader(_source: &str, _preamble_lines: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn validate_custom_shader(source: &str, preamble_lines: usize) -> Result<(), String> {
    if let Ok(path) = std::env::var("THREERS_DUMP_SHADER") {
        let _ = std::fs::write(&path, source);
    }
    let module = match wgpu::naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => return Err(annotate(&e.emit_to_string(source), source, preamble_lines)),
    };
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(_) => Ok(()),
        Err(e) => Err(annotate(&e.emit_to_string(source), source, preamble_lines)),
    }
}

/// What a custom shader's entry points reach, for a backend whose custom
/// layout cannot hold everything at once (see `custom_lit_pipeline_layout`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CustomShaderReach {
    /// A shadow map or the area-light table: group 3, outside the trimmed set.
    full_env: bool,
    /// A volume slot.
    volumes: bool,
}

/// Which of those `source` touches. naga's own usage analysis, followed through
/// helper functions, so it agrees with the check wgpu makes of the pipeline
/// against its layout. No naga means WebGPU in a browser, which never trims.
#[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
fn custom_shader_reach(source: &str) -> CustomShaderReach {
    use wgpu::naga::valid::{Capabilities, GlobalUse, ValidationFlags, Validator};
    let mut reach = CustomShaderReach::default();
    let Ok(module) = wgpu::naga::front::wgsl::parse_str(source) else {
        return reach;
    };
    let Ok(info) = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)
    else {
        return reach;
    };
    for (handle, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        let used = (0..module.entry_points.len())
            .any(|i| info.get_entry_point(i)[handle] != GlobalUse::empty());
        match (binding.group, binding.binding) {
            (3, 2..=7 | 14) => reach.full_env |= used,
            (1, 13..=15) => reach.volumes |= used,
            _ => {}
        }
    }
    reach
}

#[cfg(not(any(not(target_arch = "wasm32"), feature = "webgl")))]
fn custom_shader_reach(_source: &str) -> CustomShaderReach {
    CustomShaderReach::default()
}

/// Rewrite naga's `wgsl:<n>` positions into the caller's own numbering.
///
/// The assembled module is the built-in preamble plus the user's source, so
/// every line number naga reports is off by the length of the preamble — about
/// seventeen hundred lines of it. Unmapped, the number is not merely useless,
/// it points at a file the user does not have.
#[cfg(not(target_arch = "wasm32"))]
fn annotate(message: &str, source: &str, preamble_lines: usize) -> String {
    let total = source.lines().count();
    let mut out = String::from(message);
    out.push_str(&format!(
        "\n\nline numbers above are in the assembled module ({total} lines): the \
         built-in preamble is the first {preamble_lines}, so subtract that for a \
         position in your own source. Set THREERS_DUMP_SHADER=/tmp/shader.wgsl \
         to write the whole module out."
    ));
    out
}

/// Compile a custom-shader pipeline (alpha-blended, no depth write — treated as
/// transparent for correct compositing).
fn build_custom_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    fragment: &str,
    vertex: Option<&str>,
    format: wgpu::TextureFormat,
    transparent: bool,
    screen_space: bool,
    side: u32,
    instanced: bool,
    samples: u32,
    alpha_to_coverage: bool,
    pipes: &mut super::pipeline_state::StatePipelines,
) -> Result<Pipeline, String> {
    // Alpha-to-coverage needs samples to cover; single-sampled, the shader's
    // `coverage_alpha` cuts at half instead. And not with a shader that
    // writes its samples itself (`frag_out`, where `@builtin(sample_mask)`
    // exists): WebGPU allows one or the other, and it has said already.
    let writes_mask = samples > 1
        && crate::renderer::caps::supports_storage_buffers(device)
        && sample_mask_is_cheap()
        && fragment.contains("ThreersFragOut");
    let coverage = alpha_to_coverage && samples > 1 && !writes_mask;
    let source = custom_shader_module_source(
        fragment,
        vertex,
        crate::renderer::caps::supports_storage_buffers(device),
        coverage,
        samples,
    );
    validate_custom_shader(&source, custom_preamble_lines())?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("threers custom shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let vertex_buffers = [wgpu::VertexBufferLayout {
        array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 8,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    }];
    // `buffers` takes a slice of `Option` in wgpu 30.
    let [base] = vertex_buffers;
    // On an `InstancedMesh`: the instance matrix and tint in slot 1, as the
    // built-in instanced pipeline has them, and the two custom columns in
    // slot 2 (`InstancedCustomIn`).
    let instance_attrs =
        [(0, 4), (16, 5), (32, 6), (48, 7), (64, 9)].map(|(offset, shader_location)| {
            wgpu::VertexAttribute {
                offset,
                shader_location,
                format: wgpu::VertexFormat::Float32x4,
            }
        });
    let custom_attrs = [(0, 10), (16, 11)].map(|(offset, shader_location)| wgpu::VertexAttribute {
        offset,
        shader_location,
        format: wgpu::VertexFormat::Float32x4,
    });
    let vertex_buffers: Vec<Option<wgpu::VertexBufferLayout>> = if instanced {
        vec![
            Some(base),
            Some(wgpu::VertexBufferLayout {
                array_stride: 80,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &custom_attrs,
            }),
        ]
    } else {
        vec![Some(base)]
    };
    Ok(pipes.create(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("threers custom pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &module,
                // The built-in entry stays `vs_main`; a material that brings its own
                // declares `vs_custom`, so both can live in the module and the
                // custom one can start from `standard_vertex`.
                entry_point: Some(if vertex.is_some() {
                    "vs_custom"
                } else if instanced {
                    "vs_instanced_custom"
                } else {
                    "vs_main"
                }),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Transparent: alpha-blend, no depth write (sorted back-to-front).
                    // Opaque: replace + depth write, so the surface self-occludes.
                    // Screen-space surfaces composite over the already-blitted opaque
                    // scene, so they blend — but they still write depth, so anything
                    // drawn later sorts against the surface rather than through it.
                    blend: Some(if transparent || screen_space {
                        wgpu::BlendState::ALPHA_BLENDING
                    } else if coverage {
                        // The alpha went to coverage; what is stored is opaque —
                        // the blend constant, set white where these draw. (Only
                        // then: blending costs a tiled GPU even when it is moot.)
                        wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Constant,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }
                    } else {
                        wgpu::BlendState::REPLACE
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // three.js `side`: 0 FrontSide, 1 BackSide, 2 DoubleSide. Previously
            // hardcoded to front, which silently culls anything meant to be seen from
            // the inside — a sky dome, or a water surface viewed from below it.
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                match side {
                    1 => Some(wgpu::Face::Front),
                    2 => None,
                    _ => Some(wgpu::Face::Back),
                },
            ),
            // The material's `depthWrite` (and the rest of its `RenderState`) is
            // applied over this as a state variant at draw time.
            depth_stencil: Some(depth_stencil(
                screen_space || !transparent,
                wgpu::CompareFunction::LessEqual,
            )),
            multisample: wgpu::MultisampleState {
                count: samples,
                alpha_to_coverage_enabled: coverage,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        },
    ))
}

/// Whether a fragment shader may write `@builtin(sample_mask)` without being
/// shaded once per sample. Natively (naga → MSL, SPIR-V, HLSL) it is just an
/// output: a fur shader under 4× MSAA cost 1.4× its single-sampled self.
/// Safari's WebGPU shades such a shader per sample — a heavy fragment ran at
/// 17 fps against 59 without the output — so in the browser a shader's
/// samples go out as alpha-to-coverage instead (the share of samples
/// covered, which the hardware places: grainier, still well ahead of one
/// sample).
const fn sample_mask_is_cheap() -> bool {
    !cfg!(target_arch = "wasm32")
}

/// How many lines of built-in preamble sit above a custom shader's own source.
pub(crate) fn custom_preamble_lines() -> usize {
    custom_shader_module_source("", None, true, false, 1)
        .lines()
        .count()
}

/// Pack a flat `f32` array into an `Rgba32Float` texture, four values a texel.
///
/// This is the storage array for devices that have no storage buffers — WebGL2
/// being all of them. A `ShaderMaterial` whose data does not fit the 16 `vec4`
/// uniform block had nowhere else to put it there, which ruled the feature out
/// for anything data-driven on the web.
///
/// `textureLoad` needs no filtering and no sampler, so this works on the
/// baseline float-texture support every WebGL2 context has.
fn upload_data_texture(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Texture {
    let texels = data.len().div_ceil(4).max(1);
    // 2048 is the smallest `max_texture_dimension_2d` anything still shipping
    // reports, so a row never has to be checked against the device.
    let width = texels.clamp(1, 2048) as u32;
    let height = (texels as u32).div_ceil(width).max(1);
    let mut padded = vec![0.0f32; (width * height * 4) as usize];
    padded[..data.len()].copy_from_slice(data);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("threers user data texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&padded),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    tex
}

/// Build the custom @group(1) bind group: the mesh uniform (binding 0, shared
/// with the standard path), the `ShaderMaterial` user uniform (binding 1), the
/// three storage arrays (bindings 2-4) and the same three as textures
/// (bindings 10-12) for devices with no storage buffers. Returns the bind group
/// plus the resources it owns, which the caller keeps alive until submit.
#[allow(clippy::too_many_arguments)]
fn build_user_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgl: &wgpu::BindGroupLayout,
    mesh_uniform: wgpu::BufferBinding<'_>,
    data: &[[f32; 4]; 16],
    storage: &[Vec<f32>; 3],
    textures: &[Option<&wgpu::TextureView>],
    placeholder: &wgpu::TextureView,
    data_placeholder: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    volumes: &[crate::marks::gpu::TextureView],
    volume_placeholder: &wgpu::TextureView,
    volume_sampler: &wgpu::Sampler,
    with_volumes: bool,
) -> (wgpu::BindGroup, Vec<wgpu::Buffer>, Vec<wgpu::Texture>) {
    use wgpu::util::DeviceExt;
    let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
    let mut buffers = vec![
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers user uniform"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::UNIFORM,
        }),
    ];
    let storage_buffers = crate::renderer::caps::supports_storage_buffers(device);
    if storage_buffers {
        for s in storage.iter() {
            buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("threers user storage"),
                    contents: bytemuck::cast_slice(s),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
        }
    }
    let tex = |i: usize| -> &wgpu::TextureView {
        textures.get(i).copied().flatten().unwrap_or(placeholder)
    };
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(mesh_uniform),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: buffers[0].as_entire_binding(),
        },
    ];
    if storage_buffers {
        for (i, buffer) in buffers[1..4].iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: buffer.as_entire_binding(),
            });
        }
    }
    entries.extend([
        wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::TextureView(tex(0)),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(tex(1)),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(tex(2)),
        },
        wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(tex(3)),
        },
        wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(sampler),
        },
    ]);
    // The data textures are only built where they are the only option; with
    // storage buffers present these bind a 1x1 placeholder nothing reads, so
    // the layout is the same shape on every backend and the module is not.
    let mut data_textures: Vec<wgpu::Texture> = Vec::new();
    let mut data_views: Vec<wgpu::TextureView> = Vec::new();
    if !storage_buffers {
        for s in storage.iter() {
            let t = upload_data_texture(device, queue, s);
            data_views.push(t.create_view(&wgpu::TextureViewDescriptor::default()));
            data_textures.push(t);
        }
    }
    for i in 0..3 {
        entries.push(wgpu::BindGroupEntry {
            binding: 10 + i as u32,
            resource: wgpu::BindingResource::TextureView(
                data_views.get(i).unwrap_or(data_placeholder),
            ),
        });
    }
    if with_volumes {
        for i in 0..2 {
            entries.push(wgpu::BindGroupEntry {
                binding: 13 + i as u32,
                resource: wgpu::BindingResource::TextureView(
                    // A Metal view (a volume made for the native backend)
                    // has nothing to bind here.
                    volumes
                        .get(i)
                        .and_then(|v| v.as_wgpu())
                        .unwrap_or(volume_placeholder),
                ),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 15,
            resource: wgpu::BindingResource::Sampler(volume_sampler),
        });
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("threers custom mesh+user bg"),
        layout: bgl,
        entries: &entries,
    });
    (bg, buffers, data_textures)
}

impl Renderer {
    /// The near and far planes of the camera this renderer last drew with.
    ///
    /// Anything that linearises the depth buffer afterwards needs exactly
    /// these, and nothing else does. Defaults to `(0.1, 1000.0)` before the
    /// first frame, which is only ever read by a post pass configured and then
    /// never rendered.
    pub fn camera_planes(&self) -> (f32, f32) {
        self.last_camera_planes
    }

    /// `(tan(fov/2), aspect)` of the camera this renderer last drew with,
    /// recovered from its projection matrix. Meaningless for an orthographic
    /// camera, which has no field of view to speak of.
    pub fn camera_lens(&self) -> (f32, f32) {
        self.last_camera_lens
    }

    /// Shared device handle. Used by callers (e.g. async readback) that need
    /// to encode commands or allocate buffers outside the renderer's own loop.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }
    /// Shared queue handle (same use case as `device_arc`).
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    /// threers's reflective transparency on Standard and Physical (off by
    /// default): a transparent surface's alpha raised by its specular
    /// reflectance, `opacity + (1 − opacity) · reflectance`, so its reflection
    /// — which is in front of it and not transmitted — keeps its strength.
    /// three.js blends the whole colour by `opacity`, which fades a highlight
    /// on glass faked with `alphaMode: BLEND` into grey film.
    pub fn set_reflective_transparency(&mut self, on: bool) {
        self.reflective_transparency = on;
    }

    pub fn reflective_transparency(&self) -> bool {
        self.reflective_transparency
    }

    /// How many layers overlapping refraction glass (`TransparencyMode::Refract`)
    /// is composited in. The refracting objects are drawn back to front in
    /// up to this many groups, each refracting a capture of everything drawn
    /// before it — so glass seen through glass shows; within a group, a
    /// pixel shows only the front-most surface. Each layer past the first
    /// costs a capture (a copy and a mip chain) and a back-face pass. 2 by
    /// default; 1 is three.js's single capture.
    pub fn set_refraction_layers(&mut self, layers: usize) {
        self.refraction_layers = layers.max(1);
    }

    pub fn refraction_layers(&self) -> usize {
        self.refraction_layers
    }

    /// Enable/disable temporal anti-aliasing. When enabled, the renderer jitters
    /// the projection each frame and blends against a reprojected history buffer.
    /// Only active on the `render_to` path (needs a sampleable color target).
    /// The depth buffer the last `render` wrote, for passes that run after it.
    ///
    /// Not the offscreen target's depth: with MSAA the renderer takes a
    /// surface-style path and writes its own, leaving a target's depth texture
    /// untouched. A pass that read that one would sample an empty buffer and
    /// produce nothing, silently — which is the whole reason this accessor
    /// exists rather than callers reaching for the obvious field.
    /// Inverse view-projection of the frame just drawn, and the view-projection
    /// of the one before it — for passes that reproject after the render.
    pub fn reprojection(&self) -> ([f32; 16], [f32; 16]) {
        (self.mb_inv_vp, self.mb_prev_vp)
    }

    pub fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth_texture
    }

    /// What the last [`render`](Self::render) drew into, for a pass drawn over
    /// the frame that depth-tests against it — [`marks`](crate::marks) use
    /// this. `(multisampled colour, depth, samples)`: the colour is `Some` only
    /// when that frame went through MSAA, and must then be drawn into and
    /// resolved to the frame; otherwise the frame view itself is the colour
    /// and the depth is the renderer's own, single-sampled.
    ///
    /// After [`render_to`](Self::render_to) the frame's depth is the target's
    /// own `depth_view`, not this.
    pub fn overlay_attachments(&self) -> (Option<&wgpu::TextureView>, &wgpu::TextureView, u32) {
        match (&self.msaa_targets, self.overlay_msaa) {
            (Some((_, _, color, depth)), true) => (Some(color), depth, self.msaa),
            _ => (None, &self.depth_view, 1),
        }
    }

    /// Whether the temporal pass is enabled.
    pub fn taa(&self) -> bool {
        self.taa_enabled
    }

    pub fn set_taa(&mut self, enabled: bool) {
        if !enabled {
            self.taa_frame = 0;
            self.taa_history = None;
        }
        self.taa_enabled = enabled;
    }

    /// Accumulate a running mean of jittered frames instead of blending with a
    /// fixed feedback weight.
    ///
    /// Turn this on when the camera and the scene are both **held still** and
    /// the renderer is being asked for one converged image — an offline frame,
    /// a product shot, a thumbnail. Each call to `render` then adds one more
    /// jittered sample to the mean, so `n` renders give the quality of `n`
    /// samples a pixel, and the frame target holds the result after every one
    /// of them. Call [`reset_taa`](Self::reset_taa) before each new pose.
    ///
    /// The point of it over [`supersample`](crate::renderer::HeadlessBuilder::supersample)
    /// is memory. Supersampling renders at `size x factor`, which at 8K runs
    /// out of texture long before it runs out of usefulness — 3x of 4320x7680
    /// asks for 23040x12960. This holds one buffer at output size no matter how
    /// many samples are taken, so the same convergence is available at any
    /// resolution. The pixel throughput is the same either way; what changes is
    /// what has to exist at once.
    ///
    /// Leave it off for anything animated: without the neighbourhood clamp a
    /// moving image smears.
    pub fn set_taa_converge(&mut self, converge: bool) {
        self.taa_converge = converge;
    }

    /// Whether the TAA pass is accumulating a running mean.
    pub fn taa_converge(&self) -> bool {
        self.taa_converge
    }

    /// Whether to copy the accumulated image back into the frame target at the
    /// end of each render.
    ///
    /// Only the last sample of an accumulation run needs it: the mean lives in
    /// the history buffer, and every copy before the final one is a
    /// full-resolution blit of an image nobody reads. At 8K that is 133
    /// megapixels moved twice per sample.
    /// [`HeadlessRenderer::render_accumulated`](super::HeadlessRenderer::render_accumulated)
    /// manages this; set it by hand only if you are driving the loop yourself,
    /// and put it back before the sample you intend to read.
    pub fn set_taa_blit(&mut self, on: bool) {
        self.taa_blit = on;
    }

    /// Jitter the projection by `(x, y)` pixels on the next render, instead of
    /// by the Halton sequence, and whether or not TAA is on.
    ///
    /// For a caller accumulating frames itself: the jitter is the only part of
    /// the temporal machinery it needs, and it wants to choose the offsets.
    /// `None` restores the sequence.
    pub fn set_jitter_override(&mut self, offset: Option<(f32, f32)>) {
        self.jitter_override = offset;
    }

    /// Reset TAA accumulation (call on a hard cut / scene change to avoid ghosting).
    pub fn reset_taa(&mut self) {
        self.taa_frame = 0;
    }

    /// Turn frustum culling off, drawing every mesh whether the camera can see
    /// it or not.
    ///
    /// On by default. The only reason to switch it off is a geometry whose
    /// vertex positions are not where its buffer says they are — the renderer
    /// already exempts skinned, instanced and custom-shader draws, so this is
    /// an escape hatch rather than a knob worth turning.
    pub fn set_frustum_culling(&mut self, on: bool) {
        self.frustum_culling = on;
    }

    pub fn frustum_culling(&self) -> bool {
        self.frustum_culling
    }

    /// `(drawn, culled)` meshes for the most recent `render` call.
    ///
    /// `culled` counts draws the camera frustum rejected. They were still
    /// submitted to the shadow passes — an off-screen caster shadows the view
    /// — so this is a count of skipped *colour* draws, not of skipped work
    /// altogether.
    pub fn cull_stats(&self) -> (usize, usize) {
        self.cull_stats
    }

    /// Initial pipelines a lazy renderer has not built yet; 0 for an eager
    /// one, and once [`warm_up`](Self::warm_up) has run.
    pub fn pending_pipelines(&self) -> usize {
        self.state_pipelines.source().uncompiled().len()
    }

    /// Every initial pipeline not built yet, as a job that builds them —
    /// on this thread or others; see [`PipelineWarmup`](super::PipelineWarmup).
    /// Empty for an eager renderer.
    ///
    /// This covers the catalogue the constructor would have built, and any
    /// state variants already asked for. Variants a material's state will
    /// ask for are made at draw time: [`warm_up_states`](Self::warm_up_states)
    /// names them up front (any target), and [`warm_scene`](Self::warm_scene)
    /// draws a scene to find them (native).
    pub fn warm_up(&self) -> super::PipelineWarmup {
        super::PipelineWarmup::new(self.state_pipelines.source().uncompiled())
    }

    /// [`warm_up`](Self::warm_up), with the pipelines `rank` scores lowest
    /// built first — so a viewport that starts drawing before the job
    /// finishes has the ones it needs earliest. `rank` sees each pipeline's
    /// label (`"threers mesh pipeline"`); ties keep the order they were made in.
    pub fn warm_up_ranked(&self, rank: impl Fn(&str) -> i32) -> super::PipelineWarmup {
        let mut pipelines = self.state_pipelines.source().uncompiled();
        pipelines.sort_by_key(|p| rank(p.label().unwrap_or("")));
        super::PipelineWarmup::new(pipelines)
    }

    /// [`warm_up`](Self::warm_up) for the pipelines drawing with each of
    /// `states` will need, as well as the stock ones: one variant of every
    /// initial pipeline per state. Unlike [`warm_scene`](Self::warm_scene)
    /// this needs no scene and runs on the web. Variants a state does not
    /// change are the stock pipeline, so listing the defaults adds nothing.
    /// An eager renderer builds them here, on this thread.
    pub fn warm_up_states(
        &mut self,
        states: &[crate::materials::RenderState],
    ) -> super::PipelineWarmup {
        self.state_pipelines
            .variants_of_initial(&self.device, states);
        self.warm_up()
    }

    /// Pipelines a warm-up built and validation rejected. They stay here, so
    /// a later check sees what the job reported;
    /// [`PipelineWarmup::errors`](super::PipelineWarmup::errors) is the same
    /// list for one job.
    pub fn pipeline_failures(&self) -> Vec<super::PipelineError> {
        self.state_pipelines
            .source()
            .failed()
            .into_iter()
            .map(|(label, message)| super::PipelineError { label, message })
            .collect()
    }

    /// [`warm_up`](Self::warm_up), only for the pipelines whose label `keep`
    /// accepts (`"threers mesh pipeline"`, say) — the paths an application
    /// actually draws with.
    pub fn warm_up_matching(&self, keep: impl Fn(&str) -> bool) -> super::PipelineWarmup {
        let pipelines = self.state_pipelines.source().uncompiled();
        super::PipelineWarmup::new(
            pipelines
                .into_iter()
                .filter(|p| keep(p.label().unwrap_or("")))
                .collect(),
        )
    }

    /// The labels of the initial pipelines not built yet, for choosing what
    /// [`warm_up_matching`](Self::warm_up_matching) keeps.
    pub fn pending_pipeline_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .state_pipelines
            .source()
            .uncompiled()
            .iter()
            .map(|p| p.label().unwrap_or("").to_owned())
            .collect();
        labels.sort();
        labels.dedup();
        labels
    }

    /// Draw `scene` once into a throwaway target and wait for the GPU, so
    /// every pipeline it needs — material-state variants included — exists
    /// before the first frame anyone sees. Same size and format as this
    /// renderer's real target; nothing is presented. **Native only.**
    #[cfg(not(target_arch = "wasm32"))]
    pub fn warm_scene(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let (width, height) = self.drawing_buffer_size();
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers warm-up target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        self.render(scene, camera, &view, false);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// The cache this renderer builds its pipelines through, if it was given
    /// one ([`RendererOptions::pipeline_cache`]).
    pub fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.state_pipelines.source().cache()
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if (width, height) != self.depth_size && width > 0 && height > 0 {
            let (tex, view) = create_depth_view(&self.device, width, height);
            self.depth_texture = tex;
            self.depth_view = view;
            self.depth_size = (width, height);
            // Recreated on the next MSAA pass at the new size.
            self.msaa_targets = None;
            self.msaa_targets_spare = None;
        }
    }

    /// three.js `setViewport`: draw the next renders into this rectangle of
    /// the target. **Bottom-left origin, logical pixels** (multiplied by
    /// [`pixel_ratio`](Self::set_pixel_ratio)), as in three.js. The rectangle
    /// may hang off the target; the visible part is drawn with the projection
    /// adjusted so the picture does not squash. The default is the whole
    /// target; [`reset_viewport`](Self::reset_viewport) goes back to it.
    ///
    /// Applies to [`render`](Self::render), [`render_to`](Self::render_to) and
    /// [`render_to_multisampled`](Self::render_to_multisampled) alike — to
    /// whatever target the next render draws into — but never to cube-camera
    /// faces. See [`super::viewport`] for what a sub-rectangle turns off.
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.viewport_state.settings.viewport = Some(super::ViewportRect::new(x, y, width, height));
    }

    /// The viewport set by [`set_viewport`](Self::set_viewport), `None` for the
    /// whole target.
    pub fn viewport(&self) -> Option<super::ViewportRect> {
        self.viewport_state.settings.viewport
    }

    /// Back to drawing over the whole target.
    pub fn reset_viewport(&mut self) {
        self.viewport_state.settings.viewport = None;
    }

    /// three.js `setScissor`. Same coordinates as
    /// [`set_viewport`](Self::set_viewport); only takes effect while
    /// [`set_scissor_test`](Self::set_scissor_test) is on. With the test on and
    /// no rectangle set, the scissor is the whole target.
    pub fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.viewport_state.settings.scissor = Some(super::ViewportRect::new(x, y, width, height));
    }

    pub fn scissor(&self) -> Option<super::ViewportRect> {
        self.viewport_state.settings.scissor
    }

    /// Forget the scissor rectangle: with the test on, scissor the whole target.
    pub fn reset_scissor(&mut self) {
        self.viewport_state.settings.scissor = None;
    }

    /// three.js `setScissorTest`. While on, the colour clear and every draw are
    /// confined to the scissor rectangle — which is what lets several renders
    /// share one target without each clearing the others away.
    pub fn set_scissor_test(&mut self, enabled: bool) {
        self.viewport_state.settings.scissor_test = enabled;
    }

    pub fn scissor_test(&self) -> bool {
        self.viewport_state.settings.scissor_test
    }

    /// three.js `setPixelRatio`, for the viewport and scissor only: their
    /// rectangles are multiplied by this to get device pixels. It does not
    /// resize anything — [`resize`](Self::resize) still takes device pixels.
    /// [`HeadlessRenderer`](super::HeadlessRenderer) sets it to its supersample
    /// factor, so rectangles stay in output pixels.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.viewport_state.settings.set_pixel_ratio(ratio);
    }

    pub fn pixel_ratio(&self) -> f32 {
        self.viewport_state.settings.pixel_ratio()
    }

    /// The drawing buffer in logical pixels — device size over
    /// [`pixel_ratio`](Self::pixel_ratio). three.js `getSize`.
    pub fn size(&self) -> (f32, f32) {
        let pr = self.viewport_state.settings.pixel_ratio();
        (self.depth_size.0 as f32 / pr, self.depth_size.1 as f32 / pr)
    }

    /// The drawing buffer in device pixels. three.js `getDrawingBufferSize`.
    pub fn drawing_buffer_size(&self) -> (u32, u32) {
        self.depth_size
    }

    /// three.js `autoClear`. On (the default), each render clears colour to the
    /// scene background — the scissor rectangle when the scissor test is on,
    /// the whole target otherwise. Off, colour is kept and the scene draws over
    /// it. Depth is cleared either way.
    pub fn set_auto_clear(&mut self, enabled: bool) {
        self.viewport_state.settings.auto_clear = enabled;
    }

    pub fn auto_clear(&self) -> bool {
        self.viewport_state.settings.auto_clear
    }

    /// Enable/disable hardware MSAA on the **direct-to-surface** forward pass.
    /// `samples <= 1` turns it off; anything else selects `MSAA_SAMPLES` (4×).
    /// Render-to-texture, shadows, and scenes using glass/OIT/refraction stay
    /// single-sampled (they transparently fall back).
    pub fn set_msaa(&mut self, samples: u32) {
        let s = if samples > 1 { MSAA_SAMPLES } else { 1 };
        if s != self.msaa {
            self.msaa = s;
            self.msaa_targets = None;
            self.msaa_targets_spare = None;
        }
    }
    pub fn msaa(&self) -> u32 {
        self.msaa
    }

    /// `(passes, caster draws)`: how many shadow-map passes have been drawn
    /// and how many casters drawn into them, since the renderer was made. A
    /// pass whose light and casters have not changed is skipped, and a caster
    /// outside a pass's light frustum is not drawn into it — this says how
    /// much that saved.
    pub fn shadow_stats(&self) -> (u64, u64) {
        self.shadow_stats
    }

    /// How shadow maps are filtered — three.js's `renderer.shadowMap.type`.
    /// A light's `ShadowSettings::filter` overrides it for that light.
    pub fn set_shadow_type(&mut self, kind: crate::lights::ShadowType) {
        self.shadow_type = kind;
    }

    pub fn shadow_type(&self) -> crate::lights::ShadowType {
        self.shadow_type
    }

    /// Take WebGL2's path to a single-sampled depth under MSAA — the opaque
    /// scene's depth redrawn, rather than the multisampled one resolved —
    /// on any device. For testing that path where it is not the default.
    #[doc(hidden)]
    pub fn force_depth_replay(&mut self, on: bool) {
        self.force_depth_replay = on;
    }

    /// Whether the last render was multisampled. MSAA asked for
    /// ([`set_msaa`](Self::set_msaa)) is not always MSAA delivered: a frame
    /// drawn into a float target, or with a live depth to read back
    /// ([`render_to`](Self::render_to) draws a render target that way), or
    /// holding draws that have no multisampled pipeline (custom shaders,
    /// instanced and skinned meshes) is single-sampled. Refraction glass is
    /// multisampled on every backend, WebGL2 included.
    pub fn last_frame_multisampled(&self) -> bool {
        self.last_frame_msaa
    }

    /// Set the tone-mapping curve and exposure multiplier.
    ///
    /// Off by default so existing scenes render byte-identically. Turn it on
    /// whenever the scene carries an HDR environment, otherwise everything
    /// above 1.0 clips flat:
    ///
    /// ```no_run
    /// # use threers::renderer::ToneMapping;
    /// # fn f(renderer: &mut threers::Renderer) {
    /// renderer.set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    /// # }
    /// ```
    pub fn set_tone_mapping(&mut self, mode: ToneMapping, exposure: f32) {
        self.tone_mapping = mode;
        self.exposure = exposure;
    }

    pub fn tone_mapping(&self) -> (ToneMapping, f32) {
        (self.tone_mapping, self.exposure)
    }

    /// What the view's depth buffer is cleared to: the far plane.
    fn depth_clear(&self) -> f32 {
        if self.state_pipelines.reversed() {
            0.0
        } else {
            1.0
        }
    }

    /// Reversed depth: the view's depth runs 1 at the near plane to 0 at
    /// the far one, cleared to 0, nearer passing `Greater`.
    ///
    /// Standard `f32` depth crowds its precision against the near plane, so
    /// with a near plane of a metre the depth buffer cannot tell apart
    /// surfaces tens of metres apart a hundred kilometres away — a planet
    /// seen at an angle z-fights across whole mountains. Reversed, the
    /// float's own spacing cancels the projection's and precision is nearly
    /// even with distance.
    ///
    /// Off by default. Shadow maps stay standard. Everything that reads the
    /// view's depth back — post effects through `apply_postfx`, glass and
    /// screen-space reflection, TAA — converts it, so they see what they did
    /// before; [`Ssao`](super::ssao::Ssao), which takes the depth texture
    /// itself, needs [`Ssao::set_reversed_depth`](super::ssao::Ssao::set_reversed_depth).
    /// [`depth_texture`](Self::depth_texture) holds reversed values: `1 - d`
    /// is the standard depth.
    pub fn set_reversed_z(&mut self, on: bool) {
        self.state_pipelines.set_reversed(on);
    }

    pub fn reversed_z(&self) -> bool {
        self.state_pipelines.reversed()
    }

    /// Geometric specular antialiasing on Standard and Physical materials
    /// (off by default).
    ///
    /// Off, roughness gains three.js r165's `geometryRoughness` — the largest
    /// screen-space change in the surface normal — as three.js renders it. On,
    /// it gains the variance of the *shading* normal across the pixel instead
    /// (Kaplanyan et al. 2016, as Filament does it), which also absorbs a
    /// normal map's variance: crinkled foil or a fine normal map stops
    /// sparkling and crawling under motion, and highlights no longer match
    /// three.js's.
    pub fn set_specular_antialiasing(&mut self, on: bool) {
        self.specular_antialiasing = on;
    }

    pub fn specular_antialiasing(&self) -> bool {
        self.specular_antialiasing
    }

    /// GPU milliseconds in the main pass, from the most recently completed
    /// readback — a few frames old, and 0.0 when the adapter has no timestamp
    /// support or none has landed yet.
    ///
    /// This is the number to trust when comparing renderer changes. Wall-clock
    /// time round `render` measures CPU submission and whatever else the
    /// machine is doing; three identical frames timed that way here came back
    /// at 18, 31 and 35 ms.
    pub fn gpu_frame_ms(&self) -> f32 {
        self.timer.as_ref().map(|t| t.last_ms()).unwrap_or(0.0)
    }

    /// GPU milliseconds from the start of the first shadow pass to the end of
    /// the last, from the most recent completed readback of a render that drew
    /// them — like [`gpu_frame_ms`](Self::gpu_frame_ms), a few renders old,
    /// and 0.0 without timestamp support.
    pub fn gpu_shadow_ms(&self) -> f32 {
        self.shadow_timer
            .as_ref()
            .map(|t| t.last_ms())
            .unwrap_or(0.0)
    }

    /// three.js `renderer.shadowMap`: `auto_update` (default on) redraws the
    /// shadow maps whenever a caster or a light changed since they were last
    /// drawn — a still scene under a moving camera keeps them; turn it off
    /// and set `needs_update` to have the next render redraw them once. See
    /// [`ShadowMap`](super::ShadowMap).
    pub fn shadow_map(&self) -> &super::viewport::ShadowMap {
        &self.shadow_map
    }

    pub fn shadow_map_mut(&mut self) -> &mut super::viewport::ShadowMap {
        &mut self.shadow_map
    }

    /// How many renders so far drew the shadow maps.
    pub fn shadow_map_updates(&self) -> u64 {
        self.shadow_updates
    }
}

/// `n` sub-pixel jitter offsets in `[-0.5, 0.5)`, each axis exactly stratified.
///
/// [`halton`] is the right sequence when the sample count is not known ahead of
/// time, which is the streaming TAA case: it is spread at every prefix length.
/// When the count *is* known — an offline frame asking for exactly `n` samples —
/// it is the wrong one, because its one-dimensional marginals are only uniform
/// in the limit and at any particular `n` the two bases disagree about how
/// uniform they are. At sixteen samples base 2 lands on the exact sixteenths and
/// base 3 does not, so x and y converge at different rates. Measured as how
/// precisely a straight edge is placed as it slides across a pixel:
///
/// ```text
///                vertical edge   horizontal edge
/// Halton              0.0017 px         0.0253 px
/// stratified          0.0017 px         0.0017 px
/// ```
///
/// A renderer whose horizontal edges are fifteen times less precisely placed
/// than its vertical ones is not one anyone would ask for, and nothing about
/// the picture says which axis to distrust.
///
/// The arrangement here is N-rooks: both axes get the same ladder of `n` evenly
/// spaced positions, so both marginals are exact and both are zero-mean —
/// Halton's are not, and a jitter set whose mean is not zero translates the
/// whole frame by that much. The pairing between the two ladders is Halton's
/// own, taken by *ranking* its values rather than using them, which keeps the
/// two-dimensional spread and keeps the progressive property with it: any
/// prefix of the result is still well distributed, because ranking does not
/// reorder.
///
/// ```
/// # use threers::stratified_jitter;
/// let j = stratified_jitter(16);
/// let mean_x: f32 = j.iter().map(|o| o.0).sum::<f32>() / 16.0;
/// assert!(mean_x.abs() < 1e-6);
/// ```
pub fn stratified_jitter(n: u32) -> Vec<(f32, f32)> {
    let n = n.max(1);
    let ladder = |base: u32| -> Vec<f32> {
        let vals: Vec<f32> = (0..n).map(|k| halton(k + 1, base)).collect();
        let mut order: Vec<u32> = (0..n).collect();
        order.sort_by(|&a, &b| vals[a as usize].total_cmp(&vals[b as usize]));
        let mut out = vec![0.0; n as usize];
        for (rung, &k) in order.iter().enumerate() {
            out[k as usize] = (rung as f32 + 0.5) / n as f32 - 0.5;
        }
        out
    };
    ladder(2).into_iter().zip(ladder(3)).collect()
}

/// Halton low-discrepancy sequence — the standard source of TAA sub-pixel jitter.
///
/// `index` is one-based; `base` should be a prime, and bases 2 and 3 are the
/// conventional pair for the two screen axes. Public because
/// [`Renderer::set_jitter_override`] hands the jitter to the caller, and a
/// caller driving it wants the same sequence the renderer would have used:
///
/// ```
/// # use threers::halton;
/// let offset = (halton(4, 2) - 0.5, halton(4, 3) - 0.5); // sample 4, in [-0.5, 0.5)
/// # let _ = offset;
/// ```
pub fn halton(mut index: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// `PrimitiveState` with the fields that never vary across our pipelines filled
/// in (Ccw front, fill, clipped, non-conservative) — only topology + cull differ.
fn primitive(topology: wgpu::PrimitiveTopology, cull: Option<wgpu::Face>) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: cull,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
    }
}

/// The mesh triangle pipeline (`code`: 0 culled, 1 alpha-blended, 2 double-sided) with
/// three.js's `polygonOffset` as a depth bias.
///
/// `glPolygonOffset(factor, units)` offsets depth by `factor · m + units · r`
/// (`m` the largest window-space depth slope, `r` the smallest resolvable
/// step), and WebGPU's `DepthBiasState` is defined by the same formula with
/// `slope_scale` and `constant` — so the values carry over unchanged. Depth
/// here is not reversed (cleared to 1, `LessEqual`), so a negative offset
/// pulls toward the camera exactly as in WebGL.
#[allow(clippy::too_many_arguments)]
fn build_biased_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    samples: u32,
    code: u8,
    factor: f32,
    units: i32,
    pipes: &mut super::pipeline_state::StatePipelines,
) -> Pipeline {
    let alpha = code == 1;
    let vertex_buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 8,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    })];
    let mut ds = depth_stencil(!alpha, wgpu::CompareFunction::LessEqual);
    ds.bias = wgpu::DepthBiasState {
        constant: units,
        slope_scale: factor,
        clamp: 0.0,
    };
    pipes.create(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("threers polygon-offset pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(if alpha {
                        wgpu::BlendState::ALPHA_BLENDING
                    } else {
                        wgpu::BlendState::REPLACE
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                if code == 2 {
                    None
                } else {
                    Some(wgpu::Face::Back)
                },
            ),
            depth_stencil: Some(ds),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        },
    )
}

/// `DepthStencilState` at [`DEPTH_FORMAT`] with default stencil/bias — only the
/// write flag and compare function differ across our pipelines.
fn depth_stencil(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn create_depth_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("threers depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // COPY_SRC so the opaque depth can be copied into the sampleable ss_depth
        // for the screen-space glass ray march.
        // TEXTURE_BINDING as well: passes that run after the frame — ambient
        // occlusion, and anything else reconstructing geometry from depth —
        // have to sample this, and a depth buffer that can only be rendered
        // into and copied out of is no use to them.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// three.js's `RectAreaLight` lookup tables — `RectAreaLightUniformsLib`'s
/// `LTC_MAT_1` and `LTC_MAT_2` (Heitz et al.'s fitted linearly transformed
/// cosines), extracted from three.js r165 — as the two layers of one 64×64
/// half-float texture: half floats filter on every backend (three.js falls
/// back to the same `LTC_HALF` tables), and two layers keep the bilinear
/// filter from reading across from one table into the other.
///
/// Built with `--cfg threers_no_rect_area_light` the tables are not embedded
/// (they are 128 KB of floats that barely compress — a fifth of a small wasm)
/// and the texture is zero-filled instead: same shape, same bindings, and a
/// RectAreaLight simply contributes nothing. For consumers that never use one.
fn upload_ltc(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    #[cfg(not(threers_no_rect_area_light))]
    let half: Vec<u16> = {
        const LTC_1: &[u8] = include_bytes!("ltc/ltc_float_1.bin");
        const LTC_2: &[u8] = include_bytes!("ltc/ltc_float_2.bin");
        [LTC_1, LTC_2]
            .iter()
            .flat_map(|b| b.as_chunks::<4>().0)
            .map(|c| {
                super::gpu_texture::f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            })
            .collect()
    };
    #[cfg(threers_no_rect_area_light)]
    let half: Vec<u16> = vec![0; 64 * 64 * 4 * 2];
    let size = wgpu::Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 2,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("threers RectAreaLight LTC tables"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64 * 8),
            rows_per_image: Some(64),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// A frame bind group: the frame uniforms and a transmission capture view.
fn frame_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    capture: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(capture),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn transform_direction(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

fn mat3_to_mat4_array(m: &Matrix3) -> [f32; 16] {
    let e = &m.elements;
    [
        e[0], e[1], e[2], 0.0, e[3], e[4], e[5], 0.0, e[6], e[7], e[8], 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preamble a `ShaderMaterial` is compiled against is "everything in
    /// `SHADER_SOURCE` above the first `@fragment`", and the documented
    /// contract is that every built-in helper is in it. Those two facts are
    /// only compatible while no other fragment entry point is declared ahead of
    /// `fs_main` — and when one was, the preamble silently lost 900 lines and
    /// every shader material that called `framebuffer_encode` stopped
    /// compiling, including the one in `examples/shader_material.rs`.
    ///
    /// Nothing about that failure looks like an ordering problem: the error is
    /// `no definition in scope for identifier`, at pipeline-creation time, in
    /// the user's own shader.
    #[test]
    fn the_custom_preamble_carries_the_documented_helpers() {
        let module =
            custom_shader_module_source("@fragment\nfn fs_main() {}\n", None, true, false, 1);
        let preamble = module.split("struct ThreersUserData").next().unwrap();
        for name in [
            "fn framebuffer_encode",
            "fn linear_to_srgb",
            "fn aces_tonemap",
            "fn sample_env_cube",
            "fn pbr_brdf",
            "fn apply_fog",
            "struct VsOut",
            "fn vs_main",
        ] {
            assert!(
                preamble.contains(name),
                "`{name}` is missing from the custom-shader preamble — something \
                 declared an entry point above it in SHADER_SOURCE"
            );
        }
    }

    /// The same invariant stated directly, so a violation names the cause
    /// rather than the symptom.
    #[test]
    fn no_fragment_entry_point_precedes_fs_main() {
        let source = crate::renderer::shader::legacy_shader_source();
        let first = source
            .find("@fragment")
            .expect("SHADER_SOURCE has a fragment entry point");
        let after = &source[first..];
        let name = after
            .split("fn ")
            .nth(1)
            .and_then(|s| s.split(|c: char| !c.is_alphanumeric() && c != '_').next())
            .unwrap_or("");
        assert_eq!(
            name, "fs_main",
            "the first `@fragment` in SHADER_SOURCE is `{name}`, not `fs_main`; \
             move it below `fs_main` or it truncates the shader-material preamble"
        );
    }
}

/// Draw `frame` into a canvas of `size`, made once and kept in `slot`, and
/// hand it back: how a WebCodecs frame reaches a WebGL2 texture.
#[cfg(target_arch = "wasm32")]
fn frame_canvas(
    slot: &mut Option<(
        web_sys::HtmlCanvasElement,
        web_sys::CanvasRenderingContext2d,
    )>,
    frame: &web_sys::VideoFrame,
    size: (u32, u32),
) -> Option<web_sys::HtmlCanvasElement> {
    use wasm_bindgen::JsCast;
    if slot.is_none() {
        let canvas: web_sys::HtmlCanvasElement = web_sys::window()?
            .document()?
            .create_element("canvas")
            .ok()?
            .dyn_into()
            .ok()?;
        let ctx: web_sys::CanvasRenderingContext2d =
            canvas.get_context("2d").ok()??.dyn_into().ok()?;
        *slot = Some((canvas, ctx));
    }
    let (canvas, ctx) = slot.as_ref()?;
    if (canvas.width(), canvas.height()) != size {
        canvas.set_width(size.0);
        canvas.set_height(size.1);
    }
    // `drawImage` takes any CanvasImageSource, a VideoFrame among them.
    let draw = js_sys::Reflect::get(ctx, &"drawImage".into()).ok()?;
    let draw: &js_sys::Function = draw.unchecked_ref();
    draw.call3(ctx, frame, &0.into(), &0.into()).ok()?;
    Some(canvas.clone())
}
