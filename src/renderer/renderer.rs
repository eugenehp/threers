//! wgpu scene renderer: pipelines, resource caches, shadows, and post-processing.
//!
//! Walks a [`Scene`](crate::Scene), batches draw calls by material and topology,
//! and writes to a swapchain view or [`super::RenderTarget`]. Post-fx passes
//! share WGSL in [`super::shader`] (`POSTFX_SHADER`, effect kinds 0–30).

use bytemuck::{Pod, Zeroable};
use crate::lights::shadow::{cascade_frusta, shadow_layer_size, Caster, MAX_SHADOW_RECORDS};
use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_mesh::{geom_cache_key, GpuMesh};
use super::gpu_timer::GpuTimer;
use super::gpu_texture::{
    cube_cache_key, tex_cache_key, GpuCubeTexture, GpuTexture, MipGenerator, TexCacheKey,
};
/// Identity of the texture set a bind group binds.
///
/// `external_rt_id` is folded in because RT-backed textures bypass the texture
/// cache entirely and resolve through `rt_view_cache`, so two textures with the
/// same upload key can still bind different views.
/// The second half is the morph texture bound at binding 10 — the geometry's
/// cache key and version, `(0, 0)` for the empty default.
/// The last part is the shared sampler's [`sampler_index`].
/// Then the geometry's uv-set texture ([`Renderer::uv_key`]).
type TexBgKey = ([Option<(TexCacheKey, u32)>; 8], (usize, u32), usize, (usize, u32));

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
            let xform = [r0[0].to_bits(), r0[1].to_bits(), r0[2].to_bits(), r1[0].to_bits(), r1[1].to_bits(), r1[2].to_bits()];
            (tex_cache_key(t), t.external_rt_id.unwrap_or(u32::MAX), sampler, xform)
        })
        .collect();
    let env = slots.env_map.as_ref().map_or(0, |c| std::sync::Arc::as_ptr(c) as usize);
    (v, maps, morph, uniforms.map(|r| r.map(f32::to_bits)), env)
}

fn tex_slot_key(slot: &Option<std::sync::Arc<crate::Texture>>) -> Option<(TexCacheKey, u32)> {
    slot.as_ref()
        .map(|t| (tex_cache_key(t), t.external_rt_id.unwrap_or(u32::MAX)))
}

fn tex_bg_key(slots: &MaterialTextureSlots) -> TexBgKey {
    ([
        tex_slot_key(&slots.map),
        tex_slot_key(&slots.normal_map),
        tex_slot_key(&slots.roughness_map),
        tex_slot_key(&slots.metalness_map),
        tex_slot_key(&slots.ao_map),
        tex_slot_key(&slots.emissive_map),
        tex_slot_key(&slots.matcap_map),
        tex_slot_key(&slots.cloud_shadow_map),
    ], (0, 0), slots.sampler_source().map_or(SAMPLER_DEFAULT, |t| sampler_index(t)), (0, 0))
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
    let filter = |nearest: bool| if nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
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
        mipmap_filter: if mip == 0 { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
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
fn take_sized<T>(current: &mut Option<(u32, u32, T, T)>, spare: &mut Option<(u32, u32, T, T)>, w: u32, h: u32) -> bool {
    let fits = |t: &Option<(u32, u32, T, T)>| matches!(t, Some((cw, ch, ..)) if *cw == w && *ch == h);
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
        rot: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    };

    /// `cube`, uploaded as `gpu`, at `intensity` and `rotation`.
    fn of(cube: &CubeTexture, gpu: &GpuCubeTexture, intensity: f32, rotation: crate::math::Euler) -> Self {
        let params = match &cube.cube_uv_atlas {
            Some(a) => [a.texel_width, a.texel_height, a.lod_max as f32, 1.0],
            None => [0.0, 0.0, gpu.texture.mip_level_count().saturating_sub(1) as f32, 0.0],
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

fn material_env_key(env: (&std::sync::Arc<CubeTexture>, f32, crate::math::Euler)) -> MaterialEnvKey {
    let (cube, intensity, r) = env;
    let order = crate::math::EulerOrder::ALL.iter().position(|o| *o == r.order).unwrap_or(0) as u32;
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
    obj.clip_plane
        .map_or([0.0; 4], |p| [p.normal.x, p.normal.y, p.normal.z, p.constant])
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
    variants: &HashMap<super::material_maps::MapVariant, (wgpu::ShaderModule, wgpu::BindGroupLayout)>,
    device: &wgpu::Device,
    main: &wgpu::ShaderModule,
    base: &wgpu::RenderPipeline,
    v: Option<super::material_maps::MapVariant>,
) -> wgpu::RenderPipeline {
    let Some(v) = v else {
        return base.clone();
    };
    let Some((module, bgl)) = variants.get(&v) else {
        log::error!("threers: a draw names material variant {v:?}, which was not built");
        return base.clone();
    };
    pipes.relayout(device, base, v.key(), 2, bgl, main, module).unwrap_or_else(|| {
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
    dir: wgpu::RenderPipeline,
    spot: wgpu::RenderPipeline,
    dir_skinned: wgpu::RenderPipeline,
    spot_skinned: wgpu::RenderPipeline,
    dir_instanced: wgpu::RenderPipeline,
    spot_instanced: wgpu::RenderPipeline,
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

/// How the renderer compresses linear HDR radiance into displayable range.
///
/// Defaults to [`ToneMapping::None`], which matches the renderer's historical
/// behaviour — values above 1.0 hard-clip. That is fine for LDR content but
/// destroys hue in highlights: a clipped gold specular goes white, not gold.
/// Any scene using an HDR environment wants [`ToneMapping::AcesFilmic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneMapping {
    /// No curve; linear values are clamped to [0, 1] at output.
    #[default]
    None,
    /// Plain exposure scale, then clamp.
    Linear,
    /// ACES filmic approximation — rolls highlights off smoothly and keeps
    /// their hue, which is what makes bright metal read as metal.
    AcesFilmic,
    /// Khronos PBR Neutral — what glTF viewers use, and what to reach for when
    /// the authored colour is the point.
    ///
    /// Below the shoulder it is **almost the identity**: it subtracts a small
    /// black offset and otherwise leaves the value alone (0.18 comes out at
    /// 0.14, 0.45 at 0.432), so a material's colour survives to the screen
    /// instead of being restyled. [`AcesFilmic`](Self::AcesFilmic) lifts the
    /// same midtones hard — 0.18 to 0.267 — which is flattering on a lit scene
    /// and wrong on a colour chart.
    ///
    /// Only the top is compressed, and it desaturates as it compresses, so an
    /// over-exposed highlight resolves towards white the way an over-exposed
    /// highlight does. ACES, as the Narkowicz fit used here, is a *per-channel*
    /// curve with no cross-channel term at all, so a blown red stays a blown
    /// red rather than going white.
    Neutral,
    /// AgX, as three.js ships it.
    ///
    /// A long shoulder — it holds 3.0 down at 0.82 where ACES is already at
    /// 0.95 — so a very bright environment keeps detail in its highlights
    /// instead of flaring to a flat patch. It pays for that with its shadows,
    /// which it desaturates noticeably: a near-black saturated colour comes
    /// back closer to grey than it went in. That is the AgX look, not a defect.
    AgX,
}

impl ToneMapping {
    /// The mode index the WGSL side switches on.
    pub(crate) fn shader_mode(self) -> f32 {
        match self {
            ToneMapping::None => 0.0,
            ToneMapping::Linear => 1.0,
            ToneMapping::AcesFilmic => 2.0,
            ToneMapping::Neutral => 3.0,
            ToneMapping::AgX => 4.0,
        }
    }

    /// The same curves on the CPU, for the renderers that have no shader to run
    /// them in (`SvgRenderer`) and the path tracer's film. Takes an *exposed*
    /// linear colour, like the WGSL `apply_tone_mapping` does after it scales.
    pub fn apply(self, c: [f32; 3]) -> [f32; 3] {
        match self {
            ToneMapping::None | ToneMapping::Linear => c,
            ToneMapping::AcesFilmic => aces_cpu(c),
            ToneMapping::Neutral => neutral_cpu(c),
            ToneMapping::AgX => agx_cpu(c),
        }
    }
}

/// three.js r165 `ACESFilmicToneMapping` on an exposed colour, matching
/// `aces_tonemap` in the shader: the 1/0.6 pre-scale, the input matrix, Hill's
/// RRT + ODT fit, the output matrix, then a clamp. Column-major matrices as in
/// the GLSL, so the rows below are its columns.
fn aces_cpu(c: [f32; 3]) -> [f32; 3] {
    const INPUT: [[f32; 3]; 3] = [
        [0.59719, 0.07600, 0.02840],
        [0.35458, 0.90834, 0.13383],
        [0.04823, 0.01566, 0.83777],
    ];
    const OUTPUT: [[f32; 3]; 3] = [
        [1.60475, -0.10208, -0.00327],
        [-0.53108, 1.10813, -0.07276],
        [-0.07367, -0.00605, 1.07602],
    ];
    let mul = |m: [[f32; 3]; 3], v: [f32; 3]| -> [f32; 3] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    };
    let v = mul(INPUT, [c[0] / 0.6, c[1] / 0.6, c[2] / 0.6]);
    let fit = v.map(|x| (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.432951) + 0.238081));
    mul(OUTPUT, fit).map(|x| x.clamp(0.0, 1.0))
}

/// Khronos PBR Neutral, matching `neutral_tonemap` in the shader.
fn neutral_cpu(c: [f32; 3]) -> [f32; 3] {
    const START: f32 = 0.8 - 0.04;
    const DESAT: f32 = 0.15;
    let x = c[0].min(c[1]).min(c[2]);
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let mut c = [c[0] - offset, c[1] - offset, c[2] - offset];
    let peak = c[0].max(c[1]).max(c[2]);
    if peak < START {
        return c;
    }
    let d = 1.0 - START;
    let new_peak = 1.0 - d * d / (peak + d - START);
    for v in c.iter_mut() {
        *v *= new_peak / peak.max(1e-6);
    }
    let g = 1.0 - 1.0 / (DESAT * (peak - new_peak) + 1.0);
    [
        c[0] + (new_peak - c[0]) * g,
        c[1] + (new_peak - c[1]) * g,
        c[2] + (new_peak - c[2]) * g,
    ]
}

/// AgX, matching `agx_tonemap` in the shader. Column-major 3x3s, applied as
/// `m * v`, so the rows below are the shader's columns.
fn agx_cpu(c: [f32; 3]) -> [f32; 3] {
    fn mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    }
    const TO_2020: [[f32; 3]; 3] = [
        [0.6274, 0.0691, 0.0164],
        [0.3293, 0.9195, 0.0880],
        [0.0433, 0.0113, 0.8956],
    ];
    const TO_SRGB: [[f32; 3]; 3] = [
        [1.6605, -0.1246, -0.0182],
        [-0.5876, 1.1329, -0.1006],
        [-0.0728, -0.0083, 1.1187],
    ];
    const INSET: [[f32; 3]; 3] = [
        [0.856_627_2, 0.137_318_97, 0.111_898_21],
        [0.095_121_24, 0.761_242, 0.076_799_42],
        [0.048_251_607, 0.101_439_04, 0.811_302_4],
    ];
    const OUTSET: [[f32; 3]; 3] = [
        [1.127_100_6, -0.141_329_76, -0.141_329_76],
        [-0.110_606_64, 1.157_823_7, -0.110_606_64],
        [-0.016_493_939, -0.016_493_939, 1.251_936_4],
    ];
    const MIN_EV: f32 = -12.473_93;
    const MAX_EV: f32 = 4.026_069;

    let mut v = mul(INSET, mul(TO_2020, c));
    for x in v.iter_mut() {
        *x = ((x.max(1e-10).log2() - MIN_EV) / (MAX_EV - MIN_EV)).clamp(0.0, 1.0);
        let (x2, x4) = (*x * *x, *x * *x * *x * *x);
        *x = 15.5 * x4 * x2 - 40.14 * x4 * *x + 31.96 * x4 - 6.868 * x2 * *x
            + 0.4298 * x2
            + 0.1191 * *x
            - 0.00232;
    }
    let mut v = mul(OUTSET, v);
    for x in v.iter_mut() {
        *x = x.max(0.0).powf(2.2);
    }
    let v = mul(TO_SRGB, v);
    [
        v[0].clamp(0.0, 1.0),
        v[1].clamp(0.0, 1.0),
        v[2].clamp(0.0, 1.0),
    ]
}

/// Everything the custom pipeline descriptor branches on. Anything left out of
/// this is a material silently drawing with another material's pipeline.
/// The material's `RenderState` is not in it: that is applied over the stock
/// pipeline as a state variant (see `pipeline_state`).
type CustomPipelineKey = (u64, bool, bool, bool, u32);

/// GPU renderer: scene draws, shadow maps, environment sampling, post-fx.
pub struct Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_tri: wgpu::RenderPipeline,
    pipeline_tri_alpha: wgpu::RenderPipeline,
    // Single-layer glass (depth prepass + LessEqual color pass), sRGB + f16.
    pipeline_glass_depth: wgpu::RenderPipeline,
    pipeline_glass_color: wgpu::RenderPipeline,
    pipeline_glass_depth_f16: wgpu::RenderPipeline,
    pipeline_glass_color_f16: wgpu::RenderPipeline,
    pipeline_glass_depth_msaa: wgpu::RenderPipeline,
    pipeline_glass_color_msaa: wgpu::RenderPipeline,
    /// The tint half of the glass colour pass (`GLASS_OUTPUT` 1), drawn just
    /// before the colour half: what is behind times the glass's per-channel
    /// transmittance.
    pipeline_glass_tint: wgpu::RenderPipeline,
    pipeline_glass_tint_f16: wgpu::RenderPipeline,
    pipeline_glass_tint_msaa: wgpu::RenderPipeline,
    /// `[depth, color, tint]` glass for the encoded-blend pass under MSAA
    /// (sRGB targets only, as `encoded_blend_pipelines`).
    encoded_glass_msaa: Option<[wgpu::RenderPipeline; 3]>,
    // Weighted-blended OIT: accumulation pipeline (MRT) + resolve (sRGB + f16).
    /// Order-independent transparency accumulation. `None` where attachments
    /// cannot blend independently — see
    /// [`caps::supports_independent_blend`](crate::renderer::caps::supports_independent_blend).
    pipeline_oit: Option<wgpu::RenderPipeline>,
    /// `[tint, light]`: the OIT composite's two halves.
    pipeline_oit_resolve: [wgpu::RenderPipeline; 2],
    pipeline_oit_resolve_f16: [wgpu::RenderPipeline; 2],
    pipeline_oit_resolve_msaa: [wgpu::RenderPipeline; 2],
    /// OIT accumulation under MSAA, and its multisampled `(width, height,
    /// accum, reveal)`, resolved into `oit_targets` at the end of the pass.
    pipeline_oit_msaa: Option<wgpu::RenderPipeline>,
    oit_msaa_targets: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// As `msaa_targets_spare`, for these.
    oit_msaa_targets_spare: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// The multisampled opaque depth to the single-sampled one, for the
    /// refraction pass under MSAA (`DEPTH_RESOLVE_SHADER`). `None` where a
    /// shader cannot read a multisampled texture.
    depth_resolve: Option<(wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    /// Where depth textures cannot be copied (WebGL2): a fullscreen pass that
    /// samples one depth and writes it as another's, with its layout and the
    /// nearest sampler it reads through.
    depth_copy: Option<(wgpu::BindGroupLayout, wgpu::RenderPipeline, wgpu::Sampler)>,
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
    pipeline_tri_nocull: wgpu::RenderPipeline,
    /// Sky dome: no cull, no depth write, LessEqual (matches three.js Sky depthWrite:false).
    pipeline_tri_sky: wgpu::RenderPipeline,
    pipeline_tri_wire: wgpu::RenderPipeline,
    pipeline_line: wgpu::RenderPipeline,
    pipeline_point: wgpu::RenderPipeline,
    pipeline_point_alpha: wgpu::RenderPipeline,
    /// `LineSegments` wider than a pixel (`vs_wide_lines`), opaque and blended.
    pipeline_wide_line: wgpu::RenderPipeline,
    pipeline_wide_line_alpha: wgpu::RenderPipeline,
    pipeline_sprite: wgpu::RenderPipeline,
    // Hardware-MSAA variants of the opaque forward pipelines (surface path).
    pipeline_tri_msaa: wgpu::RenderPipeline,
    pipeline_skinned_msaa: wgpu::RenderPipeline,
    pipeline_instanced_msaa: wgpu::RenderPipeline,
    pipeline_tri_alpha_msaa: wgpu::RenderPipeline,
    pipeline_tri_nocull_msaa: wgpu::RenderPipeline,
    pipeline_tri_sky_msaa: wgpu::RenderPipeline,
    pipeline_tri_wire_msaa: wgpu::RenderPipeline,
    pipeline_line_msaa: wgpu::RenderPipeline,
    pipeline_point_msaa: wgpu::RenderPipeline,
    pipeline_wide_line_msaa: wgpu::RenderPipeline,
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
    pipeline_tri_f16: wgpu::RenderPipeline,
    pipeline_tri_alpha_f16: wgpu::RenderPipeline,
    pipeline_tri_nocull_f16: wgpu::RenderPipeline,
    pipeline_tri_sky_f16: wgpu::RenderPipeline,
    pipeline_tri_wire_f16: wgpu::RenderPipeline,
    pipeline_line_f16: wgpu::RenderPipeline,
    pipeline_point_f16: wgpu::RenderPipeline,
    pipeline_point_alpha_f16: wgpu::RenderPipeline,
    pipeline_wide_line_f16: wgpu::RenderPipeline,
    pipeline_wide_line_alpha_f16: wgpu::RenderPipeline,
    pipeline_sprite_f16: wgpu::RenderPipeline,
    pipeline_instanced_f16: wgpu::RenderPipeline,
    pipeline_skinned_f16: wgpu::RenderPipeline,
    pipeline_shadow: wgpu::RenderPipeline,
    pipeline_shadow_spot: wgpu::RenderPipeline,
    pipeline_shadow_point: wgpu::RenderPipeline,
    /// The three shadow passes for skinned casters: the posed mesh, on the
    /// skinned layout.
    pipeline_shadow_skinned: wgpu::RenderPipeline,
    pipeline_shadow_spot_skinned: wgpu::RenderPipeline,
    pipeline_shadow_point_skinned: wgpu::RenderPipeline,
    /// The three shadow passes for an `InstancedMesh` caster: the instance
    /// matrices (and per-instance morphs) from vertex slot 1.
    pipeline_shadow_instanced: wgpu::RenderPipeline,
    pipeline_shadow_spot_instanced: wgpu::RenderPipeline,
    /// The directional and spot shadow pipelines for cut-out casters.
    shadow_cut: ShadowCutPipelines,
    pipeline_shadow_point_instanced: wgpu::RenderPipeline,
    pipeline_instanced: wgpu::RenderPipeline,
    pipeline_skinned: wgpu::RenderPipeline,
    /// `wireframe: true` variants of the instanced and skinned pipelines:
    /// LineList over the mesh's wire index buffer, like `pipeline_tri_wire`.
    pipeline_instanced_wire: wgpu::RenderPipeline,
    pipeline_instanced_wire_f16: wgpu::RenderPipeline,
    /// Transparent instanced and skinned draws: alpha-blended, no depth
    /// write, as `pipeline_tri_alpha` is for plain meshes.
    pipeline_instanced_alpha: wgpu::RenderPipeline,
    pipeline_instanced_alpha_f16: wgpu::RenderPipeline,
    pipeline_skinned_alpha: wgpu::RenderPipeline,
    pipeline_skinned_alpha_f16: wgpu::RenderPipeline,
    /// The blended draws' pipelines into the `Unorm` view of an sRGB target,
    /// for the encoded-blend pass (see `encoded_pass`): sprite, transparent
    /// instanced, transparent skinned. `None` on a target that is not sRGB.
    encoded_blend_pipelines: Option<[wgpu::RenderPipeline; 8]>,
    pipeline_skinned_wire: wgpu::RenderPipeline,
    pipeline_skinned_wire_f16: wgpu::RenderPipeline,
    /// Post-fx fullscreen-quad pipeline. Reads one input texture, writes to
    /// one output attachment, applies the effect selected by the postfx
    /// uniform's effect-kind field.
    pipeline_postfx: wgpu::RenderPipeline,
    /// Additive-blend variant of the postfx pipeline. Used for bloom composite
    /// where the bloom contribution adds to the existing framebuffer content.
    pipeline_postfx_add: wgpu::RenderPipeline,
    /// Outline overlay: SrcAlpha + One (three.js AdditiveBlending).
    pipeline_postfx_outline_add: wgpu::RenderPipeline,
    /// Multiply-blend variant (DstColor * SrcColor). Used for SSAO composite.
    pipeline_postfx_multiply: wgpu::RenderPipeline,
    /// Half-float postfx variants for outline intermediates (Rgba16Float RTs).
    pipeline_postfx_f16: wgpu::RenderPipeline,
    pipeline_postfx_add_f16: wgpu::RenderPipeline,
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
    biased_pipelines: HashMap<(u8, u8, u32, i32), wgpu::RenderPipeline>,
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
    encoded_copy: Option<(wgpu::BindGroupLayout, wgpu::RenderPipeline, wgpu::RenderPipeline)>,
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
    /// Lazily-compiled custom-shader pipelines. See [`CustomPipelineKey`].
    custom_pipelines: HashMap<CustomPipelineKey, wgpu::RenderPipeline>,
    /// Transparent custom-shader pipelines into the `Unorm` view of an sRGB
    /// target, for the encoded-blend pass; keyed as `custom_pipelines` is,
    /// less the framebuffer flag.
    custom_encoded_pipelines: HashMap<(u64, bool, bool, u32), wgpu::RenderPipeline>,
    /// Custom pipelines multisampled, for an opaque shader material in a
    /// frame drawn under MSAA; keyed as `custom_encoded_pipelines` is.
    custom_msaa_pipelines: HashMap<(u64, bool, bool, u32), wgpu::RenderPipeline>,
    /// Shaders that would not compile. Remembered so the error is logged once
    /// rather than every frame, and so the draw is skipped rather than retried.
    custom_failed: std::collections::HashSet<CustomPipelineKey>,
    depth_view: wgpu::TextureView,
    /// The renderer's own depth texture (COPY_SRC) for the direct-`render` path.
    depth_texture: wgpu::Texture,
    /// 1×1 placeholder depth texture for postfx binding 6 when hw depth is unused.
    hw_depth_tex: wgpu::Texture,
    depth_size: (u32, u32),

    geom_cache: HashMap<*const BufferGeometry, CachedGpuMesh>,
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
    skin_attr_cache: HashMap<*const BufferGeometry, wgpu::Buffer>,
    /// Morph-target textures, by geometry (rebuilt on a version change).
    morph_cache: HashMap<*const BufferGeometry, MorphTexture>,
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
    uv_cache: HashMap<*const BufferGeometry, UvSetTexture>,
    /// Bound for geometry without extra uv sets (they read zero, as an
    /// attribute a geometry lacks does in three.js).
    default_uv_sets: (wgpu::Texture, wgpu::TextureView),
    /// Bound at the morph slot for every draw without morph targets.
    default_morph: (wgpu::Texture, wgpu::TextureView),
    /// Wireframe line lists for partial spans (a group or a draw range of a
    /// `wireframe: true` mesh), by (geometry, version, start, end).
    wire_span_cache: HashMap<(*const BufferGeometry, u32, u32, u32), (wgpu::Buffer, u32)>,
    /// Uploaded textures, by [`Texture::id`].
    tex_cache: HashMap<TexCacheKey, GpuTexture>,
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
    depth_replay: Vec<wgpu::RenderPipeline>,
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
    map_variants: HashMap<super::material_maps::MapVariant, (wgpu::ShaderModule, wgpu::BindGroupLayout)>,
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
    pipeline_ss_back_depth: wgpu::RenderPipeline,
    /// Fullscreen "sample one texture → write" pipeline (mip downsample + blit),
    /// one per framebuffer format.
    pipeline_ss_blit: wgpu::RenderPipeline,
    pipeline_ss_blit_f16: wgpu::RenderPipeline,
    /// Glass surface pipelines (screen-space refraction/reflection).
    pipeline_ss_glass: wgpu::RenderPipeline,
    pipeline_ss_glass_f16: wgpu::RenderPipeline,
    /// `fs_glass_shadow`: each light's shadow on the refracting surfaces,
    /// into `ss_glass_shadow`.
    pipeline_glass_shadow: wgpu::RenderPipeline,
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
    pipeline_taa: wgpu::RenderPipeline,
    pipeline_taa_f16: wgpu::RenderPipeline,

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
        let mut h = i.wrapping_mul(747_796_405).wrapping_add(salt.wrapping_mul(2_891_336_453));
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
const DEPTH_COPY_SHADER: &str = r#"
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
"#;

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
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers },
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
        Self { texture, view, size, layers }
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
        Err(e) => Err(annotate(
            &e.emit_to_string(source),
            source,
            preamble_lines,
        )),
    }
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
) -> Result<wgpu::RenderPipeline, String> {
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
    let instance_attrs = [
        (0, 4),
        (16, 5),
        (32, 6),
        (48, 7),
        (64, 9),
    ]
    .map(|(offset, shader_location)| wgpu::VertexAttribute {
        offset,
        shader_location,
        format: wgpu::VertexFormat::Float32x4,
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
    Ok(pipes.create(device, &wgpu::RenderPipelineDescriptor {
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
    }))
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
    custom_shader_module_source("", None, true, false, 1).lines().count()
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
fn upload_data_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
) -> wgpu::Texture {
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
    textures: &[std::sync::Arc<wgpu::TextureView>],
    placeholder: &wgpu::TextureView,
    data_placeholder: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    volumes: &[std::sync::Arc<wgpu::TextureView>],
    volume_placeholder: &wgpu::TextureView,
    volume_sampler: &wgpu::Sampler,
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
    let tex =
        |i: usize| -> &wgpu::TextureView { textures.get(i).map(|t| &**t).unwrap_or(placeholder) };
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
    if crate::renderer::caps::supports_shader_material_volumes(device) {
        for i in 0..2 {
            entries.push(wgpu::BindGroupEntry {
                binding: 13 + i as u32,
                resource: wgpu::BindingResource::TextureView(
                    volumes.get(i).map(|v| &**v).unwrap_or(volume_placeholder),
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

    /// Upload `geom` to the GPU now, instead of waiting for the first draw.
    ///
    /// Only needed when something outside the renderer wants the buffer up
    /// front — a compute pass that writes it, typically. See
    /// [`vertex_buffer`](Self::vertex_buffer).
    pub fn upload_geometry(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        self.ensure_geometry(geom);
    }

    /// The GPU vertex buffer backing `geom`, if it has been uploaded.
    ///
    /// Interleaved, `GpuMesh::VERTEX_STRIDE` bytes per vertex, laid out
    /// position(3), normal(3), uv(2), colour(4), tangent(4) as `f32`.
    ///
    /// Writing it from a compute pass requires
    /// [`BufferGeometry::gpu_writable`], and the handle is only valid until the
    /// geometry is re-uploaded — which happens whenever
    /// [`geometry_version`](BufferGeometry::geometry_version) changes. Drive a
    /// mesh one way or the other, not both.
    pub fn vertex_buffer(&self, geom: &std::sync::Arc<BufferGeometry>) -> Option<&wgpu::Buffer> {
        self.geom_cache
            .get(&geom_cache_key(geom))
            .map(|e| &e.mesh.vertex_buffer)
    }

    /// three.js r165's `renderTransmissionPass`, when the scene has a visible
    /// transmissive material: the opaque scene drawn linear and untone-mapped
    /// into a half-float target the size of this one, its mips filled; then,
    /// for double-sided transmissive materials, their back faces drawn over it
    /// (reading a copy of it) and the mips filled again. The frame group then
    /// binds it for the colour pass that follows.
    fn capture_transmission(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let (mut any, mut double) = (false, false);
        let mut see = |m: &Material| {
            if m.transmissive() {
                any = true;
                double |= m.side() == 2;
            }
        };
        scene.arena.traverse_visible(scene.root, &mut |_, obj| match &obj.kind {
            ObjectKind::Mesh(mesh) => mesh.draws().into_iter().for_each(|i| see(i.material)),
            ObjectKind::InstancedMesh(im) if !im.transforms.is_empty() => {
                im.draws().into_iter().for_each(|i| see(i.material))
            }
            ObjectKind::SkinnedMesh(sm) => sm.draws().into_iter().for_each(|i| see(i.material)),
            _ => {}
        });
        if !any {
            if self.transmission.take().is_some() {
                self.frame_bind_group = self.frame_bind_group_dummy.clone();
            }
            return;
        }
        let (w, h) = self.depth_size;
        if self.transmission.as_ref().map(|c| c.size) != Some((w, h)) {
            let levels = 32 - w.max(h).max(1).leading_zeros();
            let make = |label: &str, usage: wgpu::TextureUsages| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage,
                    view_formats: &[],
                })
            };
            let texture = make(
                "threers transmission capture",
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let copy = make(
                "threers transmission capture copy",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let depth = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers transmission depth"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.transmission = Some(TransmissionCapture {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                target_view: texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
                copy_view: copy.create_view(&wgpu::TextureViewDescriptor::default()),
                depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
                texture,
                copy,
                _depth: depth,
                size: (w, h),
                levels,
            });
        }
        let cap = self.transmission.as_ref().expect("just made");
        let (target, depth_view) = (cap.target_view.clone(), cap.depth_view.clone());
        let orig_depth = std::mem::replace(&mut self.depth_view, depth_view);
        let tone = std::mem::replace(&mut self.tone_map_pass, false);
        self.frame_bind_group = self.frame_bind_group_dummy.clone();

        self.capturing_transmission = Some(false);
        self.render_inner(scene, camera, &target, true, None, None);
        self.fill_capture_mips();
        if double {
            let cap = self.transmission.as_ref().expect("made above");
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers transmission copy"),
            });
            for level in 0..cap.levels {
                let size = wgpu::Extent3d {
                    width: (w >> level).max(1),
                    height: (h >> level).max(1),
                    depth_or_array_layers: 1,
                };
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &cap.texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &cap.copy,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    size,
                );
            }
            self.queue.submit([encoder.finish()]);
            self.frame_bind_group = frame_bg(
                &self.device,
                &self.frame_bgl,
                &self.frame_buffer_for_bg,
                &cap.copy_view,
                &self.transmission_sampler,
                "threers frame bg (back faces)",
            );
            let clear = std::mem::replace(&mut self.viewport_state.settings.auto_clear, false);
            self.capturing_transmission = Some(true);
            self.render_inner(scene, camera, &target, true, None, None);
            self.viewport_state.settings.auto_clear = clear;
            self.fill_capture_mips();
        }
        self.capturing_transmission = None;
        self.tone_map_pass = tone;
        self.depth_view = orig_depth;
        let cap = self.transmission.as_ref().expect("made above");
        self.frame_bind_group = frame_bg(
            &self.device,
            &self.frame_bgl,
            &self.frame_buffer_for_bg,
            &cap.view,
            &self.transmission_sampler,
            "threers frame bg",
        );
    }

    fn fill_capture_mips(&mut self) {
        let Some(cap) = &self.transmission else { return };
        let mipgen = self.mipgen.get_or_insert_with(|| MipGenerator::new(&self.device));
        mipgen.generate(&self.device, &self.queue, &cap.texture, wgpu::TextureFormat::Rgba16Float, cap.levels);
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

    fn ensure_geometry(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        let stale = self
            .geom_cache
            .get(&key)
            .map(|e| e.version != ver)
            .unwrap_or(true);
        if stale {
            let mesh = GpuMesh::upload(&self.device, geom);
            let sphere = local_bounding_sphere(geom);
            self.geom_cache.insert(
                key,
                CachedGpuMesh {
                    version: ver,
                    mesh,
                    sphere,
                },
            );
        }
        self.ensure_uv_sets(geom);
    }

    /// Upload `geom`'s `uv1`/`uv2`/`uv3` as its [`UvSetTexture`], once per
    /// geometry version; drop it when the geometry no longer has any.
    fn ensure_uv_sets(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        if self.uv_cache.get(&key).is_some_and(|u| u.version == ver) {
            return;
        }
        let sets = UV_SET_NAMES.map(|n| geom.get_attribute(n).filter(|a| a.item_size >= 2));
        let count = geom.get_attribute("position").map_or(0, |p| p.count());
        if sets.iter().all(Option::is_none) || count == 0 {
            self.uv_cache.remove(&key);
            return;
        }
        let texels = (count * 2) as u32;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = texels.min(max);
        let height = texels.div_ceil(width);
        let mut data = vec![0.0f32; (width * height * 4) as usize];
        for (set, attr) in sets.iter().enumerate() {
            let Some(a) = attr else { continue };
            for v in 0..count.min(a.count()) {
                let at = (v * 2 + set / 2) * 4 + (set % 2) * 2;
                data[at] = a.array[v * a.item_size];
                data[at + 1] = a.array[v * a.item_size + 1];
            }
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers uv sets"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.uv_cache.insert(key, UvSetTexture { version: ver, _texture: texture, view });
        self.tex_bg_cache.retain(|k, _| k.3 .0 != key as usize);
        self.variant_bg_cache.retain(|k, _| k.1 .0 != key as usize);
    }

    /// The identity of the uv-set texture a draw of geometry `key` binds —
    /// part of its bind group's cache key; `(0, 0)` for none.
    fn uv_key(&self, key: *const BufferGeometry) -> (usize, u32) {
        self.uv_cache.get(&key).map_or((0, 0), |u| (key as usize, u.version))
    }

    /// Upload `geom`'s morph targets as three.js's morph texture, once per
    /// geometry version. Returns the cache key and version to bind, or `None`
    /// when the geometry has no morph targets.
    fn ensure_morph(&mut self, geom: &std::sync::Arc<BufferGeometry>) -> Option<(usize, u32)> {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        if self.morph_cache.get(&key).is_some_and(|m| m.version == ver) {
            return Some((key as usize, ver));
        }
        let texels = crate::core::MorphTexels::build(geom)?;
        if texels.targets == 0 || texels.vertex_count == 0 {
            return None;
        }
        // three.js: one row of `vertexCount · stride` texels, wrapped onto
        // more rows when that passes the maximum texture width.
        let per_layer = texels.vertex_count * texels.stride;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = per_layer.min(max);
        let height = per_layer.div_ceil(width);
        // At least two layers: WebGL2 (wgpu's GL backend) makes a
        // one-layer texture a plain `TEXTURE_2D`, and the shader's
        // `sampler2DArray` then reads zeros from it — a single morph target
        // collapsed every vertex to the origin there. The spare layer is
        // never read.
        let layers = texels.targets.max(2);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers morph texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Each layer padded out to whole rows.
        let row = (width * 4) as usize;
        let mut padded = vec![0.0f32; row * height as usize * layers as usize];
        let layer_len = (per_layer * 4) as usize;
        for t in 0..texels.targets as usize {
            let src = &texels.data[t * layer_len..(t + 1) * layer_len];
            let dst = t * row * height as usize;
            padded[dst..dst + layer_len].copy_from_slice(src);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
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
                depth_or_array_layers: layers,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.morph_cache.insert(
            key,
            MorphTexture {
                version: ver,
                _texture: texture,
                view,
                instance_layer: None,
            },
        );
        Some((key as usize, ver))
    }

    /// An `InstancedMesh`'s morph texture: `geom`'s targets as in
    /// [`ensure_morph`](Self::ensure_morph), plus one more layer holding
    /// three.js's `InstancedMesh.morphTexture` — `instances` rows of
    /// `row_len` floats (base influence, then the influences), one float a
    /// texel in `.x`, row-major over the layer's width. Cached under `owner`
    /// (the object), rebuilt when the geometry or the row layout changes and
    /// otherwise only the rows are rewritten.
    ///
    /// One texture rather than a second binding because the mesh layout is
    /// already at WebGL2's sixteen texture units.
/// The GPU buffer for `obj`'s instance data `slot`, holding `data`: last
    /// frame's buffer when it is big enough, rewritten only if `data`
    /// changed, else a new one with room to grow.
    fn instance_buffer(
        &mut self,
        obj: crate::core::ObjectId,
        slot: u8,
        data: Vec<f32>,
        version: Option<(u64, usize)>,
    ) -> wgpu::Buffer {
        let frame = self.instance_frame;
        let bytes = (data.len() * 4) as u64;
        if let Some(c) = self.instance_cache.get_mut(&(obj, slot)) {
            if c.buffer.size() >= bytes.max(4) {
                // A versioned mesh got here because its version moved: write.
                if version.is_some() || c.data != data {
                    self.queue.write_buffer(&c.buffer, 0, bytemuck::cast_slice(&data));
                    c.data = if version.is_some() { Vec::new() } else { data };
                    self.instance_stamps += 1;
                    c.stamp = self.instance_stamps;
                }
                c.frame = frame;
                c.version = version;
                return c.buffer.clone();
            }
        }
        // Half again, so a mesh that grows a little each frame is not
        // reallocated every frame.
        let size = ((bytes + bytes / 2).max(16)).div_ceil(4) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers instance buffer"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
        self.instance_stamps += 1;
        self.instance_cache.insert(
            (obj, slot),
            InstanceBuffer {
                buffer: buffer.clone(),
                // A versioned mesh is never compared, so keep no copy.
                data: if version.is_some() { Vec::new() } else { data },
                frame,
                version,
                stamp: self.instance_stamps,
            },
        );
        buffer
    }

    /// The content stamp of `obj`'s instance slot, 0 when it has none.
    fn instance_stamp(&self, obj: crate::core::ObjectId, slot: u8) -> u64 {
        self.instance_cache.get(&(obj, slot)).map_or(0, |c| c.stamp)
    }

    /// Last frame's buffer for `obj`'s slot when its `instance_version` (and
    /// count) say nothing changed — without reading the instance data.
    fn cached_instances(
        &mut self,
        obj: crate::core::ObjectId,
        slot: u8,
        version: Option<(u64, usize)>,
    ) -> Option<wgpu::Buffer> {
        let v = version?;
        let c = self.instance_cache.get_mut(&(obj, slot))?;
        if c.version != Some(v) {
            return None;
        }
        c.frame = self.instance_frame;
        Some(c.buffer.clone())
    }

    fn ensure_instance_morph(
        &mut self,
        owner: usize,
        geom: &std::sync::Arc<BufferGeometry>,
        rows: &[f32],
        instances: u32,
        row_len: u32,
    ) -> Option<(usize, u32)> {
        let key = owner as *const BufferGeometry;
        // High bit set, so it can never equal a plain geometry version.
        let ver = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (geom_cache_key(geom), geom.geometry_version, instances, row_len).hash(&mut h);
            (h.finish() as u32) | 0x8000_0000
        };
        let texel_rows = |width: u32| -> Vec<f32> {
            let n = (instances * row_len) as usize;
            let rows_needed = (n as u32).div_ceil(width).max(1);
            let mut out = vec![0.0f32; (width * rows_needed * 4) as usize];
            for (i, v) in rows.iter().take(n).enumerate() {
                out[i * 4] = *v;
            }
            out
        };
        if let Some(m) = self.morph_cache.get(&key).filter(|m| m.version == ver) {
            let (layer, width) = m.instance_layer?;
            let data = texel_rows(width);
            let rows_needed = (data.len() / 4) as u32 / width;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &m._texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 16),
                    rows_per_image: Some(rows_needed),
                },
                wgpu::Extent3d { width, height: rows_needed, depth_or_array_layers: 1 },
            );
            return Some((owner, ver));
        }
        let texels = crate::core::MorphTexels::build(geom)?;
        if texels.targets == 0 || texels.vertex_count == 0 || instances == 0 {
            return None;
        }
        let per_layer = texels.vertex_count * texels.stride;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = per_layer.min(max);
        let height = per_layer
            .div_ceil(width)
            .max((instances * row_len).div_ceil(width));
        if height > max {
            return None;
        }
        let layers = (texels.targets + 1).max(2);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers instanced morph texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row = (width * 4) as usize;
        let layer_size = row * height as usize;
        let mut padded = vec![0.0f32; layer_size * layers as usize];
        let layer_len = (per_layer * 4) as usize;
        for t in 0..texels.targets as usize {
            let src = &texels.data[t * layer_len..(t + 1) * layer_len];
            padded[t * layer_size..t * layer_size + layer_len].copy_from_slice(src);
        }
        let inst = texel_rows(width);
        let at = texels.targets as usize * layer_size;
        padded[at..at + inst.len()].copy_from_slice(&inst);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
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
            wgpu::Extent3d { width, height, depth_or_array_layers: layers },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.morph_cache.insert(
            key,
            MorphTexture {
                version: ver,
                _texture: texture,
                view,
                instance_layer: Some((texels.targets, width)),
            },
        );
        Some((owner, ver))
    }

    /// The wireframe line list of one partial span of `geom` — a group or a
    /// draw range — cached by geometry version and span.
    fn ensure_wire_span(&mut self, geom: &std::sync::Arc<BufferGeometry>, start: u32, end: u32) {
        let k = (geom_cache_key(geom), geom.geometry_version, start, end);
        if self.wire_span_cache.contains_key(&k) {
            return;
        }
        // A runaway guard: spans are few and stable per geometry, and a
        // stale version's entries are simply never asked for again.
        if self.wire_span_cache.len() > 1024 {
            self.wire_span_cache.clear();
        }
        let flat = geom.wireframe_indices_in(start, end);
        let buf = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers wireframe span index buffer"),
                // An empty buffer cannot be bound; one dummy index, drawn zero
                // times, stands in.
                contents: bytemuck::cast_slice(if flat.is_empty() { &[0u32] } else { &flat }),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        self.wire_span_cache.insert(k, (buf, flat.len() as u32));
    }

    /// Give `geom`'s cached upload its wireframe line list. Call after
    /// [`ensure_geometry`](Self::ensure_geometry) for a `wireframe: true`
    /// material; a re-upload drops the list, so it follows `geometry_version`.
    fn ensure_wire_indices(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        if let Some(e) = self.geom_cache.get_mut(&geom_cache_key(geom)) {
            e.mesh.ensure_wire_indices(&self.device, geom);
        }
    }

    /// (Re)create the OIT accumulation + revealage targets at `w×h` and their
    /// resolve bind group. No-op if already the right size.
    /// (Re)create the mipmapped opaque-scene capture used by the screen-space
    /// glass pass. Sized `w×h` in `format` (must match the framebuffer format so
    /// the opaque pipelines can render into mip 0). The mip chain feeds the
    /// roughness→LOD blur of the refracted/reflected image.
    /// The scratch a layer of overlapping glass draws into (`ss_layer`), the
    /// capture's size and format.
    fn ensure_ss_layer(&mut self) {
        let Some(c) = self.ss_color.as_ref() else { return };
        if matches!(&self.ss_layer, Some((w, h, t, _)) if *w == c.width && *h == c.height && t.format() == c.format) {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ss glass layer"),
            size: wgpu::Extent3d { width: c.width, height: c.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: c.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ss_layer = Some((c.width, c.height, texture, view));
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

    fn ensure_ss_targets(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) {
        let (w, h) = (w.max(1), h.max(1));
        if let Some(c) = &self.ss_color {
            if c.width == w && c.height == h && c.format == format {
                return;
            }
        }
        // ~8 mips is plenty for a roughness blur; cap so the smallest is ≥1px.
        let max_mips = 32 - (w.max(h)).leading_zeros();
        let mips = max_mips.clamp(1, 8);
        let unorm = format.remove_srgb_suffix();
        let reinterpret = unorm != format && super::caps::supports_view_formats(&self.device);
        let unorm_formats = [unorm];
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ss color capture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_DST: a layer of overlapping glass is copied in (`ss_layer`).
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: if reinterpret { &unorm_formats } else { &[] },
        });
        let all_mips_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mip0_unorm = reinterpret.then(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip 0 unorm"),
                format: Some(unorm),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            })
        });
        let mut mip_render_views = Vec::with_capacity(mips as usize);
        let mut mip_sample_views = Vec::with_capacity(mips as usize);
        for m in 0..mips {
            mip_render_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip render"),
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            }));
            mip_sample_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip sample"),
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            }));
        }
        self.ss_color = Some(SsColor {
            width: w,
            height: h,
            texture,
            all_mips_view,
            mip_render_views,
            mip_sample_views,
            mip0_unorm,
            mips,
            format,
        });

        // Sampleable copy of the opaque depth (marched by the glass shader).
        let need_depth = match &self.ss_depth {
            Some((cw, ch, ..)) => *cw != w || *ch != h,
            None => true,
        };
        if need_depth {
            let dtex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ss depth copy"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                // a render target too, for where it is filled by drawing (`depth_copy`)
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_depth = Some((w, h, dtex, dview));
            // Glass back-face depth target (rendered into, then sampled).
            let btex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ss back depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let bview = btex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_back_depth = Some((w, h, btex, bview));
            let stex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers glass shadows"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GLASS_SHADOW_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let sview = stex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_glass_shadow = Some((stex, sview));
        }
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

    fn ensure_taa_targets(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) {
        let (w, h) = (w.max(1), h.max(1));
        // The format is part of the match, not just the size: turning
        // convergence on switches the history to float, and a pair of buffers
        // kept because they were merely the right *shape* would leave the
        // accumulation back in eight bits with nothing to show for it.
        if let Some((cw, ch, ..)) = &self.taa_history {
            if *cw == w && *ch == h && self.taa_hist_format == Some(format) {
                return;
            }
        }
        self.taa_hist_format = Some(format);
        // The cached views and bind groups point at the buffers about to be
        // replaced.
        self.taa_bind_cache = None;
        let mk = || {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers taa history"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (a, av) = mk();
        let (b, bv) = mk();
        self.taa_history = Some((w, h, a, av, b, bv));
        self.taa_frame = 0;
    }

    fn ensure_oit_targets(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if let Some((cw, ch, ..)) = &self.oit_targets {
            if *cw == w && *ch == h {
                return;
            }
        }
        let mk = |fmt: wgpu::TextureFormat, label: &str| {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (accum, accum_view) = mk(wgpu::TextureFormat::Rgba16Float, "threers oit accum");
        let (reveal, reveal_view) = mk(OIT_REVEAL_FORMAT, "threers oit reveal");
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers oit resolve bg"),
            layout: &self.oit_resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&reveal_view),
                },
            ],
        });
        self.oit_targets = Some((w, h, accum, accum_view, reveal, reveal_view, bg));
    }

    /// The OIT accumulation's multisampled targets, resolved into
    /// `oit_targets`.
    fn ensure_oit_msaa_targets(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if !take_sized(&mut self.oit_msaa_targets, &mut self.oit_msaa_targets_spare, w, h) {
            return;
        }
        let mk = |format: wgpu::TextureFormat, label: &str| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: MSAA_SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let accum = mk(wgpu::TextureFormat::Rgba16Float, "threers oit accum msaa");
        let reveal = mk(OIT_REVEAL_FORMAT, "threers oit reveal msaa");
        self.oit_msaa_targets = Some((w, h, accum, reveal));
    }

    /// Create pipelines and default 1×1 fallback textures.
    ///
    /// `color_format` is the swapchain or render-target format (typically
    /// `Rgba8Unorm` or `Rgba8UnormSrgb` on web, `Bgra8UnormSrgb` on some desktops).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::legacy_shader_source().into()),
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // 1–2: the transmission capture and its trilinear sampler.
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers frame bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers mesh bgl"),
            entries: &[uniform_entry(0)],
        });

        // Skinned variant of the mesh BGL: same uniform at binding 0, plus a
        // storage buffer of bone matrices at binding 1. This lets the skinned
        // pipeline stay within the 4-bind-group WebGPU default limit.
        let mesh_skinned_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers mesh skinned bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 7 fragment textures (albedo, normal, roughness, metalness, ao,
        // emissive, matcap) + 1 sampler, and for the vertex stage the morph
        // texture and the uv sets. Every other map — displacement included —
        // is a shader variant's (`material_maps`): with the frame group's
        // transmission capture and the environment group, this layout is at
        // WebGL2's sixteen texture units.
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // Binding 10: three.js's morph-target texture, read in the vertex
        // stage with `textureLoad` — unfilterable float, so `Rgba32Float`
        // works everywhere and it lowers to `texelFetch` on WebGL2. Eleven
        // textures here plus the environment group's five stays inside
        // WebGL2's sixteen units.
        let morph_entry = wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers tex bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
                tex_entry(6),
                uv_sets_entry(),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    // The displacement sample needs the same sampler, from the
                    // vertex stage.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                morph_entry,
            ],
        });

        // Bindings 9..12 are the screen-space capture the refraction pass reads.
        // They are left out where that pass cannot be compiled, and the reason is
        // the texture budget rather than tidiness: WebGL2 allots sixteen units,
        // and this group together with the nine material maps asks for
        // seventeen.
        let ltc = upload_ltc(&device, &queue);
        let env_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
                // 14: the `RectAreaLight` tables, filtered as three.js reads
                // them (`LinearFilter`).
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 13: `EnvUniforms`, this group's environment's parameters.
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
        ];
        // The main layout stops here, deliberately. Bindings 9..12 are the
        // screen-space capture, and no entry point outside the refraction pass
        // reads them — but a declared binding is not free: wgpu-hal's GL backend
        // assigns a texture unit per *layout* entry, not per use, which is why a
        // lit draw that samples none of them still ran out of units. They live
        // in a second layout below, used only by the pass that needs them.
        let env_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers env bgl"),
            entries: &env_entries,
        });

        // The refraction pass's own group 3: the environment and the capture, and
        // *not* the three shadow maps, which `fs_ss_glass` never samples. Those
        // three units are exactly what makes room for the capture's three, so
        // both layouts come to the same total and both fit inside WebGL2's
        // sixteen.
        let mut env_glass_entries: Vec<wgpu::BindGroupLayoutEntry> = env_entries
            .iter()
            .filter(|e| matches!(e.binding, 0 | 1 | 8 | 13))
            .cloned()
            .collect();
        env_glass_entries.extend([
                // 9/10: screen-space glass — mipmapped opaque-scene capture +
                // its trilinear sampler (only read by the fs_ss_glass entry).
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 11: sampleable opaque depth; 12: glass back-face depth (both
                // marched/read by fs_ss_glass via textureLoad, no sampler).
                // 1×1 placeholders bound otherwise.
                // Unfilterable *float*, not `Depth`, even though the views are
                // depth-format — a depth format permits either. The shader reads
                // these with `textureLoad`, and GLSL has no `texelFetch` for a
                // depth sampler, so declared as depth the module cannot be
                // lowered for WebGL2 at all. As float textures the same reads
                // become `texelFetch` and the pass compiles everywhere.
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 15: each light's shadow on the glass, packed by
                // `fs_glass_shadow` — the one unit left, standing in for the
                // three shadow maps.
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
        ]);
        let env_bgl_glass = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers env bgl (refraction)"),
            entries: &env_glass_entries,
        });
        let ss_glass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers ss glass layout"),
                bind_group_layouts: &[
                    Some(&frame_bgl),
                    Some(&mesh_bgl),
                    Some(&tex_bgl),
                    Some(&env_bgl_glass),
                ],
                immediate_size: 0,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&mesh_bgl), Some(&tex_bgl), Some(&env_bgl)],
            immediate_size: 0,
        });

        // Skinned pipeline layout: same as the base 4 groups, but group 1 uses
        // the skinned mesh BGL (uniform + bones storage). Stays at 4 groups —
        // WebGPU's default `maxBindGroups` limit.
        let pipeline_layout_skinned =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers skinned layout"),
                bind_group_layouts: &[Some(&frame_bgl), Some(&mesh_skinned_bgl), Some(&tex_bgl), Some(&env_bgl)],
                immediate_size: 0,
            });

        // The shadow passes' group 0: the frame uniforms and, at a dynamic
        // offset, the pass's own matrix (`ShadowPassGpu`). Their pipelines
        // get layouts of their own with it in place of the frame group.
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers shadow pass bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ShadowPassGpu>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let shadow_pass_stride = {
            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            (std::mem::size_of::<ShadowPassGpu>() as u64).div_ceil(align) * align
        };
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers shadow layout"),
            bind_group_layouts: &[Some(&shadow_pass_bgl), Some(&mesh_bgl), Some(&tex_bgl), Some(&env_bgl)],
            immediate_size: 0,
        });
        let shadow_layout_skinned = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers skinned shadow layout"),
            bind_group_layouts: &[Some(&shadow_pass_bgl), Some(&mesh_skinned_bgl), Some(&tex_bgl), Some(&env_bgl)],
            immediate_size: 0,
        });
        // Point passes: no environment group.
        let shadow_layout_point = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers point shadow layout"),
            bind_group_layouts: &[Some(&shadow_pass_bgl), Some(&mesh_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });

        // ShaderMaterial data rides in @group(1) alongside the mesh uniform
        // (WebGPU's default maxBindGroups is 4, so we can't add a 5th group):
        //   binding 0 = mesh uniform (as usual), binding 1 = user uniform block,
        //   bindings 2..4 = read-only storage arrays.
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let volume_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let custom_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // The storage arrays again, as textures, for the devices that have no
        // storage buffers. `textureLoad` needs no filtering, so these stay
        // non-filterable and work with `Rgba32Float` everywhere.
        let data_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // The storage entries are omitted where the device has none — WebGL2's
        // limit is zero, and declaring them anyway fails layout creation
        // outright, which takes the whole renderer down before a frame is drawn.
        let mut custom_entries = vec![
            uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            // The user's own data is visible to the vertex stage as well, which
            // is what makes a custom vertex stage worth having: a ribbon or a
            // strand needs its control points there, not in the fragment.
            uniform_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
        ];
        if crate::renderer::caps::supports_storage_buffers(&device) {
            custom_entries.extend([storage_entry(2), storage_entry(3), storage_entry(4)]);
        }
        custom_entries.extend([
                // User textures + their sampler. Unbound slots get a 1x1 white
                // placeholder, so a shader that ignores them costs nothing.
                custom_tex_entry(5),
                custom_tex_entry(6),
                custom_tex_entry(7),
                custom_tex_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                data_tex_entry(10),
                data_tex_entry(11),
                data_tex_entry(12),
        ]);
        // `ShaderMaterial::volumes`: two 3D textures and their clamp-to-edge
        // sampler, where they fit (not on WebGL2).
        if crate::renderer::caps::supports_shader_material_volumes(&device) {
            custom_entries.extend([
                volume_entry(13),
                volume_entry(14),
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]);
        }
        let custom_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers custom mesh+user bgl"),
            entries: &custom_entries,
        });
        // Group 2 for shader materials: the sampler and the per-geometry
        // vertex data, and nothing else.
        //
        // `ShaderMaterial` has no standard maps — it carries a fragment source,
        // a uniform block and its own textures — so the eight material textures
        // in `tex_bgl` are definitionally unused by a custom shader. Declaring
        // them anyway cost eight fragment texture units, which is what put the
        // custom layout over the per-stage limit of sixteen and made shader
        // materials unavailable on baseline WebGPU altogether.
        //
        // Binding 7 stays for the shared `vs_main`'s sampler; the morph
        // texture and uv sets are the geometry's. A `ShaderMaterial` has no
        // displacement map, so one bind group serves every custom draw.
        let custom_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers custom tex bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The shared vertex stage reads the morph texture and the uv
                // sets too; a shader material binds the geometry's or the
                // empty defaults.
                morph_entry,
                uv_sets_entry(),
            ],
        });

        // The custom pipeline's group 3. A `ShaderMaterial` does its own
        // lighting and samples neither the shadow maps nor the area-light table
        // the full environment carries, so where a backend charges a texture
        // unit per layout entry rather than per use — wgpu's GL backend, i.e.
        // WebGL2 — those are dead weight that would crowd out the volume slots:
        // the full group plus the volumes comes to eighteen there and WebGL2
        // allows sixteen, and going over panics inside wgpu-hal. On that backend
        // the pipeline takes a trimmed environment (the IBL cube, its 2D
        // fallback, the sampler and the uniform — bindings 0, 1, 8, 13), which
        // brings the layout back to fourteen. Where units are counted per use
        // (WebGPU, Metal, Vulkan) the full group costs nothing a shader does not
        // touch, so it stays and custom shaders keep shadows and area lights.
        // "No storage buffers" is the same backend, and the test used elsewhere.
        // `Some` only on the trimmed backend; elsewhere the custom pipeline
        // reuses the full `env_bgl` *object*, so the scene's `env_bind_group`
        // (built from it) binds to it without a separate compatible layout.
        let custom_env_trimmed = !crate::renderer::caps::supports_storage_buffers(&device);
        let custom_env_bgl = custom_env_trimmed.then(|| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = env_entries
                .iter()
                .filter(|e| matches!(e.binding, 0 | 1 | 8 | 13))
                .cloned()
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers custom env bgl (trimmed)"),
                entries: &entries,
            })
        });

        // Built only when the device can hold it.
        //
        // The four groups together once declared 20 sampled textures in the
        // fragment stage. WebGPU's floor is 16 and Chrome reports exactly that,
        // so that layout could not be created there at all; trimming the custom
        // tex group of the eight standard material maps a `ShaderMaterial` never
        // uses brought it to fourteen (see `custom_tex_bgl`). Below that floor —
        // genuinely old hardware — the feature is absent rather than broken, and
        // `None` is what says so. See `caps::CUSTOM_LAYOUT_TEXTURES` for the
        // count and how WebGL2's per-layout-entry budget is met.
        use crate::renderer::caps::CUSTOM_LAYOUT_TEXTURES;
        let custom_pipeline_layout = if device.limits().max_sampled_textures_per_shader_stage
            >= CUSTOM_LAYOUT_TEXTURES
        {
            Some(
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("threers custom (shader material) layout"),
                    bind_group_layouts: &[
                        Some(&frame_bgl),
                        Some(&custom_mesh_bgl),
                        Some(&custom_tex_bgl),
                        Some(custom_env_bgl.as_ref().unwrap_or(&env_bgl)),
                    ],
                    immediate_size: 0,
                }),
            )
        } else {
            log::warn!(
                "shader materials unavailable: this device allows {} sampled textures per \
                 fragment stage and the custom layout needs {CUSTOM_LAYOUT_TEXTURES}",
                device.limits().max_sampled_textures_per_shader_stage
            );
            None
        };
        // The screen-space (refraction) custom pipeline keeps the glass group 3,
        // whose capture textures leave no room for the volume slots on WebGL2
        // (nine in the mesh group with the volumes, plus the glass group, tips
        // over sixteen). A trimmed-environment backend therefore has no
        // screen-space custom pipeline; such a draw is skipped like any whose
        // shader will not compile. Screen-space refraction in a `ShaderMaterial`
        // and 3D-volume sampling in one material is a combination nothing here
        // asks for, and the per-use backends keep it.
        let custom_ss_pipeline_layout = custom_pipeline_layout
            .as_ref()
            .filter(|_| !custom_env_trimmed)
            .map(|_| {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("threers custom (screen-space shader material) layout"),
                    bind_group_layouts: &[
                        Some(&frame_bgl),
                        Some(&custom_mesh_bgl),
                        Some(&custom_tex_bgl),
                        Some(&env_bgl_glass),
                    ],
                    immediate_size: 0,
                })
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
        let vertex_buffers = vertex_buffers.map(Some);
        // `Points`: the same vertex buffer, one vertex per *instance* — each
        // point is expanded to a quad by `vs_points`.
        let vertex_buffers_points = vertex_buffers.clone().map(|b| {
            b.map(|mut b| {
                b.step_mode = wgpu::VertexStepMode::Instance;
                b
            })
        });
        // Wide lines: a segment an instance, its start through slot 0 (as
        // `VsIn`) and its end through slot 1 — the same buffer bound a vertex
        // further on — each stepping two vertices (`vs_wide_lines`).
        let wide_stride = 2 * GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress;
        let vertex_buffers_wide = [
            vertex_buffers[0].clone().map(|mut b| {
                b.step_mode = wgpu::VertexStepMode::Instance;
                b.array_stride = wide_stride;
                b
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: wide_stride,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0, shader_location: 9, format: wgpu::VertexFormat::Float32x3 },
                    wgpu::VertexAttribute { offset: 24, shader_location: 10, format: wgpu::VertexFormat::Float32x2 },
                    wgpu::VertexAttribute { offset: 32, shader_location: 11, format: wgpu::VertexFormat::Float32x4 },
                ],
            }),
        ];

        // Every pipeline a material draws with is created through this, which
        // keeps its descriptor so a material's `depthTest` / `depthWrite` /
        // `depthFunc` / `colorWrite` can be applied to it later — see
        // `pipeline_state`.
        let state_pipes =
            std::cell::RefCell::new(super::pipeline_state::StatePipelines::default());
        let record = |desc: &wgpu::RenderPipelineDescriptor| {
            state_pipes.borrow_mut().create(&device, desc)
        };
        // The layouts whose group 2 is the material's maps, registered so a
        // pipeline on them can be rebuilt for a material shader variant
        // (`material_maps`, `StatePipelines::relayout`).
        {
            let mut sp = state_pipes.borrow_mut();
            sp.register_layout(
                &pipeline_layout,
                vec![Some(frame_bgl.clone()), Some(mesh_bgl.clone()), Some(tex_bgl.clone()), Some(env_bgl.clone())],
            );
            sp.register_layout(
                &pipeline_layout_skinned,
                vec![Some(frame_bgl.clone()), Some(mesh_skinned_bgl.clone()), Some(tex_bgl.clone()), Some(env_bgl.clone())],
            );
            sp.register_layout(
                &shadow_layout,
                vec![Some(shadow_pass_bgl.clone()), Some(mesh_bgl.clone()), Some(tex_bgl.clone()), Some(env_bgl.clone())],
            );
            sp.register_layout(
                &shadow_layout_skinned,
                vec![Some(shadow_pass_bgl.clone()), Some(mesh_skinned_bgl.clone()), Some(tex_bgl.clone()), Some(env_bgl.clone())],
            );
            sp.register_layout(
                &shadow_layout_point,
                vec![Some(shadow_pass_bgl.clone()), Some(mesh_bgl.clone()), Some(tex_bgl.clone())],
            );
            sp.register_layout(
                &ss_glass_pipeline_layout,
                vec![Some(frame_bgl.clone()), Some(mesh_bgl.clone()), Some(tex_bgl.clone()), Some(env_bgl_glass.clone())],
            );
        }

        let make_pipeline = |fmt: wgpu::TextureFormat,
                             topology: wgpu::PrimitiveTopology,
                             cull: Option<wgpu::Face>,
                             label: &str| {
            let alpha = label.contains("alpha");
            // Note: PolygonMode::Line needs the POLYGON_MODE_LINE feature, which
            // is generally unavailable in browser WebGPU. The wireframe pipeline
            // instead uses LineList primitives over the triangle indices (fed as
            // edge indices at draw time), so every pipeline stays Fill.
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if alpha {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                // Transparent: don't write depth, so back fragments behind a
                // transparent surface still composite correctly. LessEqual is
                // three.js's default `depthFunc`.
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Single-layer "glass" pipelines. `GLASS_DEPTH` (the prepass) writes
        // the nearest glass depth with no color; the tint and light halves of
        // the colour pass (`GLASS_OUTPUT` in the shader) blend only the
        // fragments at that nearest depth (LessEqual) and don't write depth:
        // the tint multiplies what is behind by the glass's transmittance, the
        // light adds its reflection, highlights and emission.
        let make_glass_pipeline = |fmt: wgpu::TextureFormat, pass: u8, count: u32, label: &str| {
            let depth_only = pass == GLASS_DEPTH;
            let blend = match pass {
                GLASS_TINT => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::Src,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                GLASS_LIGHT => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                _ => wgpu::BlendState::ALPHA_BLENDING,
            };
            let constants: &[(&str, f64)] = match pass {
                GLASS_TINT => &[("GLASS_OUTPUT", 1.0)],
                GLASS_LIGHT => &[("GLASS_OUTPUT", 2.0)],
                _ => &[],
            };
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(blend),
                        write_mask: if depth_only {
                            wgpu::ColorWrites::empty()
                        } else {
                            wgpu::ColorWrites::ALL
                        },
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                depth_stencil: Some(depth_stencil(
                    depth_only,
                    if depth_only {
                        wgpu::CompareFunction::Less
                    } else {
                        wgpu::CompareFunction::LessEqual
                    },
                )),
                multisample: wgpu::MultisampleState {
                    count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let f16_format = wgpu::TextureFormat::Rgba16Float;
        let pipeline_glass_depth =
            make_glass_pipeline(color_format, GLASS_DEPTH, 1, "threers glass depth pipeline");
        let pipeline_glass_color =
            make_glass_pipeline(color_format, GLASS_LIGHT, 1, "threers glass color pipeline");
        let pipeline_glass_tint =
            make_glass_pipeline(color_format, GLASS_TINT, 1, "threers glass tint pipeline");
        let pipeline_glass_depth_f16 =
            make_glass_pipeline(f16_format, GLASS_DEPTH, 1, "threers glass depth f16 pipeline");
        let pipeline_glass_color_f16 =
            make_glass_pipeline(f16_format, GLASS_LIGHT, 1, "threers glass color f16 pipeline");
        let pipeline_glass_tint_f16 =
            make_glass_pipeline(f16_format, GLASS_TINT, 1, "threers glass tint f16 pipeline");
        // Single-layer glass multisampled, so a scene with glass keeps MSAA.
        let pipeline_glass_depth_msaa =
            make_glass_pipeline(color_format, GLASS_DEPTH, MSAA_SAMPLES, "threers glass depth msaa pipeline");
        let pipeline_glass_color_msaa =
            make_glass_pipeline(color_format, GLASS_LIGHT, MSAA_SAMPLES, "threers glass color msaa pipeline");
        let pipeline_glass_tint_msaa =
            make_glass_pipeline(color_format, GLASS_TINT, MSAA_SAMPLES, "threers glass tint msaa pipeline");
        // And for the encoded-blend pass into the multisampled `Unorm` view.
        let encoded_glass_msaa = color_format.is_srgb().then(|| {
            let unorm = color_format.remove_srgb_suffix();
            [
                make_glass_pipeline(unorm, GLASS_DEPTH, MSAA_SAMPLES, "threers glass depth encoded msaa pipeline"),
                make_glass_pipeline(unorm, GLASS_LIGHT, MSAA_SAMPLES, "threers glass color encoded msaa pipeline"),
                make_glass_pipeline(unorm, GLASS_TINT, MSAA_SAMPLES, "threers glass tint encoded msaa pipeline"),
            ]
        });

        // Screen-space refraction glass: opaque single-surface pass (writes depth,
        // Less test, no blend — it composites the refracted/reflected background
        // itself), sampling the captured ss-color from env-group binding 9.
        let make_ss_glass_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&ss_glass_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_ss_glass"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                // `LessEqual`: `fs_glass_shadow` has just written the front-most
                // glass depth, and this shades exactly that surface.
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Built everywhere. It has its own group 3 — the screen-space capture in
        // place of the shadow maps it never samples — so it comes to the same
        // fourteen texture units the main layout does, inside WebGL2's sixteen.
        let pipeline_ss_glass = make_ss_glass_pipeline(color_format, "threers ss glass pipeline");
        // Each light's shadow on the refracting surfaces, for the pass above
        // (`fs_glass_shadow`): the lit layout, which has the shadow maps.
        let pipeline_glass_shadow = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers glass shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_glass_shadow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GLASS_SHADOW_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Back)),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_ss_glass_f16 =
            make_ss_glass_pipeline(f16_format, "threers ss glass f16 pipeline");

        // Fullscreen "sample one texture → write" pipeline, used both to build
        // the ss-color mip chain (downsample) and to blit the capture back to the
        // frame target before the glass draws over it.
        let ss_blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ss blit bgl"),
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
            ],
        });
        let ss_blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ss blit layout"),
            bind_group_layouts: &[Some(&ss_blit_bgl)],
            immediate_size: 0,
        });
        let ss_blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ss blit shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::SS_BLIT_SHADER.into()),
        });
        let make_ss_blit_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&ss_blit_layout),
                vertex: wgpu::VertexState {
                    module: &ss_blit_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ss_blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_ss_blit = make_ss_blit_pipeline(color_format, "threers ss blit pipeline");
        let pipeline_ss_blit_f16 =
            make_ss_blit_pipeline(f16_format, "threers ss blit f16 pipeline");

        // Depth-only pass over glass back faces (cull front) → the exit depth of
        // the glass volume, for two-surface (thick) refraction. `vs_main` reads
        // the displacement map out of group 2, so this needs the texture layout
        // too — otherwise displaced glass would write back-face depth for
        // geometry that is not where it is drawn.
        let ss_back_depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ss back depth layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&mesh_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });
        state_pipes.borrow_mut().register_layout(
            &ss_back_depth_layout,
            vec![Some(frame_bgl.clone()), Some(mesh_bgl.clone()), Some(tex_bgl.clone())],
        );
        let pipeline_ss_back_depth =
            record(&wgpu::RenderPipelineDescriptor {
                label: Some("threers ss back depth pipeline"),
                layout: Some(&ss_back_depth_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Front),
                ),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // --- Temporal anti-aliasing resolve ---
        let taa_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers taa uniform"),
            size: std::mem::size_of::<TaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let taa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers taa sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tex_entry = |binding, depth| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                // Unfilterable float for the depth slot rather than `Depth`: the
                // resolve reads it with `textureLoad`, which has no GLSL
                // equivalent for a depth sampler. A depth format allows either
                // sample type, so this costs nothing and lets the pass compile
                // on WebGL2 as well.
                sample_type: if depth {
                    wgpu::TextureSampleType::Float { filterable: false }
                } else {
                    wgpu::TextureSampleType::Float { filterable: true }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let taa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers taa bgl"),
            entries: &[
                tex_entry(0, false), // current
                tex_entry(1, false), // history
                tex_entry(2, true),  // depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
        let taa_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers taa layout"),
            bind_group_layouts: &[Some(&taa_bgl)],
            immediate_size: 0,
        });
        let taa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers taa shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::TAA_SHADER.into()),
        });
        let make_taa = |fmt: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&taa_layout),
                vertex: wgpu::VertexState {
                    module: &taa_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &taa_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Built everywhere, including WebGL2: the resolve's depth slot is an
        // unfilterable float texture rather than a depth texture, so its
        // `textureLoad` lowers to `texelFetch` and the module compiles there too.
        let pipeline_taa = make_taa(color_format, "threers taa pipeline");
        let pipeline_taa_f16 = make_taa(f16_format, "threers taa f16 pipeline");

        // --- Weighted-blended OIT ---
        // Accumulation pass: fs_oit writes to accum (additive) + revealage
        // (multiplicative), depth-tested against opaque but not writing depth.
        let oit_accum_format = wgpu::TextureFormat::Rgba16Float;
        let oit_reveal_format = OIT_REVEAL_FORMAT;
        let add_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mul_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        };
        // The accumulation pass writes weighted colour and coverage to two
        // attachments with different blend equations, which WebGL2 cannot do —
        // it blends every attachment alike. Where that is the case the pipeline
        // is not built and the pass does not run; transparency falls back to the
        // sorted path.
        let make_oit = |count: u32, label: &str| {
        record(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_oit"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: oit_accum_format,
                        blend: Some(wgpu::BlendState {
                            color: add_blend,
                            alpha: add_blend,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: oit_reveal_format,
                        blend: Some(wgpu::BlendState {
                            color: mul_blend,
                            alpha: mul_blend,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                // The light half of a blend, premultiplied, and the transmittance
                // beside it (`shade_main`'s `GLASS_OUTPUT`).
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("GLASS_OUTPUT", 2.0)],
                    ..Default::default()
                },
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState { count, ..Default::default() },
            multiview_mask: None,
            cache: None,
        })
        };
        let independent_blend = crate::renderer::caps::supports_independent_blend(&device);
        let pipeline_oit = independent_blend.then(|| make_oit(1, "threers oit accum pipeline"));
        // Multisampled, against the multisampled opaque depth, resolved
        // before the composite: a frame with OIT glass keeps its MSAA.
        let pipeline_oit_msaa =
            independent_blend.then(|| make_oit(MSAA_SAMPLES, "threers oit accum msaa pipeline"));
        // Resolve pass: composite accum/revealage over the opaque image.
        let oit_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers oit resolve bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers oit resolve shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::OIT_RESOLVE_SHADER.into()),
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers oit resolve layout"),
            bind_group_layouts: &[Some(&oit_resolve_bgl)],
            immediate_size: 0,
        });
        // Two halves, as the glass colour pass is: the transmittance
        // multiplied over what is behind, then the light added.
        let make_resolve = |fmt: wgpu::TextureFormat, pass: u32, count: u32, label: &str| {
            let blend = if pass == 0 {
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::Src,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }
            } else {
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&resolve_layout),
                vertex: wgpu::VertexState {
                    module: &resolve_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("RESOLVE_PASS", pass as f64)],
                        ..Default::default()
                    },
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_oit_resolve = [
            make_resolve(color_format, 0, 1, "threers oit resolve tint pipeline"),
            make_resolve(color_format, 1, 1, "threers oit resolve light pipeline"),
        ];
        let pipeline_oit_resolve_f16 = [
            make_resolve(f16_format, 0, 1, "threers oit resolve tint f16 pipeline"),
            make_resolve(f16_format, 1, 1, "threers oit resolve light f16 pipeline"),
        ];
        // Into the multisampled colour, resolved to the frame: a pass drawn
        // over the frame afterwards (`overlay_attachments`) resolves the
        // multisampled colour again, and has to find the composite in it.
        let pipeline_oit_resolve_msaa = [
            make_resolve(color_format, 0, MSAA_SAMPLES, "threers oit resolve tint msaa pipeline"),
            make_resolve(color_format, 1, MSAA_SAMPLES, "threers oit resolve light msaa pipeline"),
        ];
        // A depth texture to another, by drawing, where they cannot be copied
        // (WebGL2): what the refraction pass's sampleable depth is filled by.
        // The depth is read through a nearest sampler, as `textureLoad` of a
        // depth texture reads 0 there.
        let depth_copy = (!crate::renderer::caps::supports_depth_copies(&device)).then(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers depth copy bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("threers depth copy shader"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_SHADER.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers depth copy layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers depth copy pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("threers depth copy sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (bgl, pipe, sampler)
        });
        // A multisampled depth to a single-sampled one, for the refraction
        // pass under MSAA. Not where shaders cannot read multisampled
        // textures (WebGL2); there a refraction frame stays single-sampled.
        let depth_resolve = crate::renderer::caps::supports_multisampled_textures(&device).then(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers depth resolve bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("threers depth resolve shader"),
                source: wgpu::ShaderSource::Wgsl(super::shader::DEPTH_RESOLVE_SHADER.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers depth resolve layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers depth resolve pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Always)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            (bgl, pipeline)
        });
        let pipeline_tri = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri pipeline",
        );
        let pipeline_tri_alpha = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha pipeline",
        );
        let pipeline_tri_nocull = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull pipeline",
        );
        let make_sky_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_tri_sky = make_sky_pipeline(color_format, "threers tri sky pipeline");
        // `wireframe: true` materials: a LineList pipeline (Fill polygon mode, so no
        // NON_FILL_POLYGON_MODE feature needed and it works in browser WebGPU) drawn
        // over the mesh's separate wire index buffer — `BufferGeometry::wireframe_indices`,
        // i.e. polygon outlines when the geometry carries a polygon table, else every
        // triangle edge as three.js draws them. See `GpuMesh::wire_index_buffer`.
        let pipeline_tri_wire = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire pipeline",
        );
        let pipeline_line = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line pipeline",
        );
        // three.js `Points`: a quad per point (`vs_points`), opaque — replace,
        // depth written — or blended, in either target format, single- or
        // multisampled.
        let make_point_pipeline = |fmt: wgpu::TextureFormat, blend: bool, samples: u32, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_points"),
                    buffers: &vertex_buffers_points,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if blend { wgpu::BlendState::ALPHA_BLENDING } else { wgpu::BlendState::REPLACE }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(!blend, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_point = make_point_pipeline(color_format, false, 1, "threers point pipeline");
        let pipeline_point_alpha = make_point_pipeline(color_format, true, 1, "threers point alpha pipeline");
        // `LineSegments` wider than a pixel: a capsule a segment
        // (`vs_wide_lines` / `fs_wide_lines`), opaque or blended, as points are.
        let make_wide_line_pipeline = |fmt: wgpu::TextureFormat, blend: bool, samples: u32, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_wide_lines"),
                    buffers: &vertex_buffers_wide,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_wide_lines"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if blend { wgpu::BlendState::ALPHA_BLENDING } else { wgpu::BlendState::REPLACE }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(!blend, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_wide_line = make_wide_line_pipeline(color_format, false, 1, "threers wide line pipeline");
        let pipeline_wide_line_alpha = make_wide_line_pipeline(color_format, true, 1, "threers wide line alpha pipeline");

        // --- Hardware-MSAA variants of the opaque forward pipelines (surface
        // path only). Identical to the ones above but with `count = MSAA_SAMPLES`;
        // used when antialiasing is enabled and the scene has no glass/OIT/refract
        // /instanced/skinned/custom draws (those fall back to the 1× path). ---
        let make_pipeline_ms = |fmt: wgpu::TextureFormat,
                                topology: wgpu::PrimitiveTopology,
                                cull: Option<wgpu::Face>,
                                label: &str| {
            let alpha = label.contains("alpha");
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if alpha {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let make_sky_pipeline_ms = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_tri_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri msaa",
        );
        let pipeline_tri_alpha_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha msaa",
        );
        let pipeline_tri_nocull_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull msaa",
        );
        let pipeline_tri_wire_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire msaa",
        );
        let pipeline_line_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line msaa",
        );
        let pipeline_point_msaa = make_point_pipeline(color_format, false, MSAA_SAMPLES, "threers point msaa");
        let pipeline_wide_line_msaa = make_wide_line_pipeline(color_format, false, MSAA_SAMPLES, "threers wide line msaa");
        let pipeline_tri_sky_msaa = make_sky_pipeline_ms(color_format, "threers tri sky msaa");

        let pipeline_tri_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri f16 pipeline",
        );
        let pipeline_tri_alpha_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha f16 pipeline",
        );
        let pipeline_tri_nocull_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull f16 pipeline",
        );
        let pipeline_tri_sky_f16 = make_sky_pipeline(f16_format, "threers tri sky f16 pipeline");
        let pipeline_tri_wire_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire f16 pipeline",
        );
        let pipeline_line_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line f16 pipeline",
        );
        let pipeline_point_f16 = make_point_pipeline(f16_format, false, 1, "threers point f16 pipeline");
        let pipeline_point_alpha_f16 =
            make_point_pipeline(f16_format, true, 1, "threers point alpha f16 pipeline");
        let pipeline_wide_line_f16 = make_wide_line_pipeline(f16_format, false, 1, "threers wide line f16 pipeline");
        let pipeline_wide_line_alpha_f16 =
            make_wide_line_pipeline(f16_format, true, 1, "threers wide line alpha f16 pipeline");

        // Instanced pipeline: vertex layout slot 0 (per-vertex), slot 1 per-instance mat4.
        let vertex_buffers_instanced = [
            wgpu::VertexBufferLayout {
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
            },
            wgpu::VertexBufferLayout {
                // Sixteen floats of matrix plus four of tint.
                array_stride: 80,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 48,
                        shader_location: 7,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 64,
                        shader_location: 9,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
        ];
        // `buffers` takes a slice of `Option` in wgpu 30.
        let vertex_buffers_instanced = vertex_buffers_instanced.map(Some);
        // Skinned vertex layout: slot 0 = standard vertex, slot 1 = joints+weights.
        let vertex_buffers_skinned = [
            wgpu::VertexBufferLayout {
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
            },
            wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
        ];
        // `buffers` takes a slice of `Option` in wgpu 30.
        let vertex_buffers_skinned = vertex_buffers_skinned.map(Some);
        // Skinned and instanced pipelines, each in a filled (back-face culled)
        // and a `wireframe: true` LineList variant, for the surface format and
        // the half-float postfx format. The two kinds differ only in layout,
        // vertex entry point and vertex buffers.
        type MeshKind<'a> = (
            &'a wgpu::PipelineLayout,
            &'a str,
            &'a [Option<wgpu::VertexBufferLayout<'a>>],
        );
        let skinned: MeshKind = (&pipeline_layout_skinned, "vs_skinned", &vertex_buffers_skinned);
        let instanced: MeshKind = (&pipeline_layout, "vs_instanced", &vertex_buffers_instanced);
        let make_mesh_pipeline_samples = |(layout, entry, buffers): MeshKind,
                                          fmt: wgpu::TextureFormat,
                                          topology: wgpu::PrimitiveTopology,
                                          cull: Option<wgpu::Face>,
                                          samples: u32,
                                          label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let make_mesh_pipeline = |kind: MeshKind,
                                  fmt: wgpu::TextureFormat,
                                  topology: wgpu::PrimitiveTopology,
                                  cull: Option<wgpu::Face>,
                                  label: &str| {
            make_mesh_pipeline_samples(kind, fmt, topology, cull, 1, label)
        };
        let tris = wgpu::PrimitiveTopology::TriangleList;
        let lines = wgpu::PrimitiveTopology::LineList;
        let back = Some(wgpu::Face::Back);
        // Multisampled, so a frame holding a skinned or instanced mesh keeps
        // its MSAA.
        let pipeline_skinned_msaa = make_mesh_pipeline_samples(
            skinned,
            color_format,
            tris,
            back,
            MSAA_SAMPLES,
            "threers skinned msaa pipeline",
        );
        let pipeline_instanced_msaa = make_mesh_pipeline_samples(
            instanced,
            color_format,
            tris,
            back,
            MSAA_SAMPLES,
            "threers instanced msaa pipeline",
        );
        let pipeline_skinned =
            make_mesh_pipeline(skinned, color_format, tris, back, "threers skinned pipeline");
        let pipeline_skinned_f16 =
            make_mesh_pipeline(skinned, f16_format, tris, back, "threers skinned f16 pipeline");
        let pipeline_skinned_wire =
            make_mesh_pipeline(skinned, color_format, lines, None, "threers skinned wire pipeline");
        let pipeline_skinned_wire_f16 = make_mesh_pipeline(
            skinned,
            f16_format,
            lines,
            None,
            "threers skinned wire f16 pipeline",
        );
        let pipeline_instanced =
            make_mesh_pipeline(instanced, color_format, tris, back, "threers instanced pipeline");
        let pipeline_instanced_f16 =
            make_mesh_pipeline(instanced, f16_format, tris, back, "threers instanced f16 pipeline");
        let pipeline_instanced_wire = make_mesh_pipeline(
            instanced,
            color_format,
            lines,
            None,
            "threers instanced wire pipeline",
        );
        let pipeline_instanced_wire_f16 = make_mesh_pipeline(
            instanced,
            f16_format,
            lines,
            None,
            "threers instanced wire f16 pipeline",
        );

        let pipeline_sprite = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers sprite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sprite"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_sprite_f16 = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers sprite f16 pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sprite"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: f16_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Alpha-blended draws on the skinned, instanced and sprite layouts:
        // the transparent instanced and skinned pipelines in both target
        // formats, and — on an sRGB target — all three into its `Unorm` view
        // for the encoded-blend pass. No depth write, as `pipeline_tri_alpha`.
        let make_blend_pipeline = |(layout, entry, buffers): MeshKind,
                                   fmt: wgpu::TextureFormat,
                                   cull: Option<wgpu::Face>,
                                   compare: wgpu::CompareFunction,
                                   label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, cull),
                depth_stencil: Some(depth_stencil(false, compare)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // three.js's default `depthFunc`.
        let less = wgpu::CompareFunction::LessEqual;
        let pipeline_instanced_alpha =
            make_blend_pipeline(instanced, color_format, back, less, "threers instanced alpha pipeline");
        let pipeline_instanced_alpha_f16 =
            make_blend_pipeline(instanced, f16_format, back, less, "threers instanced alpha f16 pipeline");
        let pipeline_skinned_alpha =
            make_blend_pipeline(skinned, color_format, back, less, "threers skinned alpha pipeline");
        let pipeline_skinned_alpha_f16 =
            make_blend_pipeline(skinned, f16_format, back, less, "threers skinned alpha f16 pipeline");
        let sprite_kind: MeshKind = (&pipeline_layout, "vs_sprite", &vertex_buffers);
        let encoded_blend_pipelines = color_format.is_srgb().then(|| {
            let unorm = color_format.remove_srgb_suffix();
            [
                make_blend_pipeline(
                    sprite_kind,
                    unorm,
                    None,
                    wgpu::CompareFunction::LessEqual,
                    "threers sprite encoded pipeline",
                ),
                make_blend_pipeline(instanced, unorm, back, less, "threers instanced encoded pipeline"),
                make_blend_pipeline(skinned, unorm, back, less, "threers skinned encoded pipeline"),
                make_point_pipeline(unorm, true, 1, "threers point encoded pipeline"),
                make_glass_pipeline(unorm, GLASS_DEPTH, 1, "threers glass depth encoded pipeline"),
                make_glass_pipeline(unorm, GLASS_LIGHT, 1, "threers glass color encoded pipeline"),
                make_glass_pipeline(unorm, GLASS_TINT, 1, "threers glass tint encoded pipeline"),
                make_wide_line_pipeline(unorm, true, 1, "threers wide line encoded pipeline"),
            ]
        });

        // Depth-only pipeline used to render the scene from each shadow-casting
        // light's POV. Reuses the per-vertex layout + per-mesh uniforms.
        let make_shadow_pipeline = |entry: &str, label: &str, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                // A cut-out caster (`alphaTest` with a map) tests the map's
                // alpha per fragment; every other caster is depth-only.
                fragment: cut.then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_cut"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Front),
                ),
                // Shadow bias reduces acne; keep the explicit DepthStencilState.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow = make_shadow_pipeline("vs_shadow", "threers shadow pipeline", false);
        let pipeline_shadow_spot =
            make_shadow_pipeline("vs_shadow_spot", "threers spot shadow pipeline", false);
        // The only shadow pipeline with a fragment stage: see `fs_shadow_point`.
        let pipeline_shadow_point = {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some("threers point shadow pipeline"),
                layout: Some(&shadow_layout_point),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_shadow_point"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_point"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Shadow casters that are skinned: the same three passes on the
        // skinned layout and vertex buffers, so the map holds the pose the
        // colour pass draws rather than the bind pose (or nothing).
        let make_skinned_shadow_pipeline = |entry: &str, label: &str, point: bool, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_layout_skinned),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers_skinned,
                    compilation_options: Default::default(),
                },
                fragment: (point || cut).then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if point { "fs_shadow_point" } else { "fs_shadow_cut" }),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: if point {
                    wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    }
                } else {
                    primitive(wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Front))
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: if point {
                        Default::default()
                    } else {
                        wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        }
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_skinned",
            "threers skinned shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_spot_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_spot_skinned",
            "threers skinned spot shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_point_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_point_skinned",
            "threers skinned point shadow pipeline",
            true,
            false,
        );

        // Instanced casters: the same three passes on the instanced vertex
        // layout. The point pass's layout is the point pipeline's — frame,
        // mesh and the morph texture's group — and the other two share the
        // mesh pipelines' layout, as their non-instanced twins do.
        let make_instanced_shadow_pipeline = |entry: &str, label: &str, point: bool, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(if point { &shadow_layout_point } else { &shadow_layout }),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers_instanced,
                    compilation_options: Default::default(),
                },
                fragment: (point || cut).then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if point { "fs_shadow_point" } else { "fs_shadow_cut" }),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: if point {
                    wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    }
                } else {
                    primitive(wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Front))
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: if point {
                        Default::default()
                    } else {
                        wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        }
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_instanced",
            "threers instanced shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_spot_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_spot_instanced",
            "threers instanced spot shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_point_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_point_instanced",
            "threers instanced point shadow pipeline",
            true,
            false,
        );

        // Cut-out casters: the directional and spot passes again, with
        // `fs_shadow_cut` testing the map's alpha against `alphaTest`.
        let shadow_cut = ShadowCutPipelines {
            dir: make_shadow_pipeline("vs_shadow", "threers cut-out shadow pipeline", true),
            spot: make_shadow_pipeline("vs_shadow_spot", "threers cut-out spot shadow pipeline", true),
            dir_skinned: make_skinned_shadow_pipeline(
                "vs_shadow_skinned",
                "threers skinned cut-out shadow pipeline",
                false,
                true,
            ),
            spot_skinned: make_skinned_shadow_pipeline(
                "vs_shadow_spot_skinned",
                "threers skinned cut-out spot shadow pipeline",
                false,
                true,
            ),
            dir_instanced: make_instanced_shadow_pipeline(
                "vs_shadow_instanced",
                "threers instanced cut-out shadow pipeline",
                false,
                true,
            ),
            spot_instanced: make_instanced_shadow_pipeline(
                "vs_shadow_spot_instanced",
                "threers instanced cut-out spot shadow pipeline",
                false,
                true,
            ),
        };

        // The opaque scene's depth again, single-sampled: for screen-space
        // refraction under MSAA where a shader cannot read the multisampled
        // depth to resolve it (WebGL2). The caster vertex stages, given the
        // camera's view-projection for their matrix; no colour, no bias.
        // `[plain, skinned, instanced]`, then the same with the cut-out
        // fragment stage for `alphaTest` with a map.
        let make_replay = |which: usize, cut: bool| {
            let (entry, layout, buffers): (&str, &wgpu::PipelineLayout, &[Option<wgpu::VertexBufferLayout>]) =
                match which {
                    0 => ("vs_shadow", &shadow_layout, &vertex_buffers[..]),
                    1 => ("vs_shadow_skinned", &shadow_layout_skinned, &vertex_buffers_skinned[..]),
                    _ => ("vs_shadow_instanced", &shadow_layout, &vertex_buffers_instanced[..]),
                };
            record(&wgpu::RenderPipelineDescriptor {
                label: Some("threers depth replay pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: cut.then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_cut"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Back)),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let depth_replay: Vec<wgpu::RenderPipeline> =
            [(0, false), (1, false), (2, false), (0, true), (1, true), (2, true)]
                .iter()
                .map(|&(w, c)| make_replay(w, c))
                .collect();

        // Post-fx fullscreen-quad pipeline. Reads one input texture, writes
        // to a render attachment matching the surface format.
        let postfx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers postfx shader"),
            source: wgpu::ShaderSource::Wgsl(crate::renderer::shader::POSTFX_SHADER.into()),
        });
        let postfx_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers postfx bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Unfilterable float rather than `Depth`, so the
                        // `textureLoad` in `sample_postfx_depth` lowers to
                        // `texelFetch`. A depth format allows either.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let postfx_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers postfx pipeline layout"),
                bind_group_layouts: &[Some(&postfx_bgl)],
                immediate_size: 0,
            });
        let pipeline_postfx = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers postfx pipeline"),
            layout: Some(&postfx_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &postfx_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &postfx_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let postfx_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let postfx_nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let postfx_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx noise sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let ssao_kernel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers ssao kernel buffer"),
            size: (32 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssao_noise_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ssao noise"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_noise_view = ssao_noise_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let glitch_disp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers glitch disp"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let glitch_disp_view = glitch_disp_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let rgba_upload_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgba upload"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rgba_upload_view = rgba_upload_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Same pipeline as `pipeline_postfx`, but with additive blending so
        // the shader output adds to existing framebuffer pixels (bloom composite).
        let pipeline_postfx_add = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers postfx additive pipeline"),
            layout: Some(&postfx_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &postfx_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &postfx_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_postfx_outline_add =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx outline additive pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let pipeline_postfx_multiply =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx multiply pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Dst,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let postfx_f16_format = wgpu::TextureFormat::Rgba16Float;
        let pipeline_postfx_f16 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers postfx f16 pipeline"),
            layout: Some(&postfx_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &postfx_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &postfx_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: postfx_f16_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_postfx_add_f16 =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx additive f16 pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: postfx_f16_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let sprite_quad = {
            let mut g = crate::core::BufferGeometry::new();
            g.set_attribute(
                "position",
                crate::core::BufferAttribute::new(
                    vec![
                        -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
                    ],
                    3,
                ),
            );
            g.set_attribute(
                "uv",
                crate::core::BufferAttribute::new(vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 2),
            );
            g.set_index(vec![0, 1, 2, 0, 2, 3]);
            GpuMesh::upload(&device, &g)
        };

        let (depth_texture, depth_view) = create_depth_view(&device, width, height);
        let hw_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers postfx hw depth placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers frame uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transmission_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers transmission sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let transmission_dummy = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers transmission stand-in"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let frame_buffer_for_bg = frame_buffer.clone();
        let frame_bind_group =
            frame_bg(&device, &frame_bgl, &frame_buffer, &transmission_dummy.1, &transmission_sampler, "threers frame bg");
        let frame_bind_group_dummy =
            frame_bg(&device, &frame_bgl, &frame_buffer, &transmission_dummy.1, &transmission_sampler, "threers frame bg (capture)");

        let sampler_linear_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers linear-repeat sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            // Anisotropic filtering. Mip-mapping alone still blurs a texture seen
            // at a grazing angle — the whole surface of a sphere near its limb —
            // because the sampler has to pick one level for a footprint that is
            // long in one direction and short in the other.
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let samplers = (0..SAMPLER_COUNT).map(|i| make_sampler(&device, i)).collect();

        let default_white_srgb = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8UnormSrgb),
        );
        let default_data_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers user data placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_data_tex_view =
            default_data_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let default_volume_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("threers user volume placeholder"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler_volume = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers volume sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let default_white_linear = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8Unorm),
        );

        // The morph slot's placeholder: one texel, one layer, never read
        // (a draw without morph targets uploads a zero count).
        let default_morph = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers default morph texture"),
                // Two layers so the GL backend makes it an array texture,
                // matching the shader's declaration (see `ensure_morph`).
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            (texture, view)
        };
        let default_uv_sets = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers default uv sets"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let custom_tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers custom tex bg"),
            layout: &custom_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&default_uv_sets.1),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear_repeat),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&default_morph.1),
                },
            ],
        });
        let default_normal = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([128, 128, 255, 255], TextureFormat::Rgba8Unorm),
        );
        let default_transparent = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([0, 0, 0, 0], TextureFormat::Rgba8Unorm),
        );

        // Default 1x1 environment cubemap: all-black faces (zero env contribution).
        let default_env_cube = {
            let face_data = [0u8; 4];
            let faces = [
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
            ];
            let src = CubeTexture::new(1, TextureFormat::Rgba8UnormSrgb, faces);
            GpuCubeTexture::upload(&device, &queue, &src)
        };
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers env sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Trilinear sampler for the screen-space glass capture — roughness maps
        // to a mip LOD so rough glass blurs its refracted/reflected image.
        let ss_glass_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers ss glass sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Shadow depth texture for the directional light's depth pre-pass.
        //
        // 4096, up from 1024. The map covers whatever the light's `camera_size`
        // spans, so its resolution is only meaningful as a ratio: a vehicle
        // 8 m across got 8 mm of shadow per texel at 1024, which is coarse
        // enough to see as staircase edges on every strut. At 4096 it is 2 mm,
        // and the cost is one 64 MB depth target and a depth pass over the same
        // geometry — nothing next to the colour pass at 8K.
        //
        // KEEP IN STEP with SHADOW_TEXEL in shader.rs, which sizes the PCF
        // kernel and cannot read this constant.
        let dir_shadows = ShadowArray::new(&device, "threers shadow depth", SPOT_SHADOW_SIZE, 1);
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let spot_shadows = ShadowArray::new(&device, "threers spot shadow depth", SPOT_SHADOW_SIZE, 1);
        let spot_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers spot shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Point shadows: six faces per light, in a 2D array.
        let point_shadows = ShadowArray::new(&device, "threers point shadow depth", POINT_SHADOW_SIZE, 6);
        let point_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers point shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let default_cube_uv_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers default cube uv"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_cube_uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_cube_uv_view =
            default_cube_uv_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let make_env_uniform = |u: EnvUniforms, label: &str| {
            wgpu::util::DeviceExt::create_buffer_init(
                &*device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            )
        };
        // The scene's, rewritten every frame; and one with no environment,
        // for the shadow passes, which read none.
        let env_uniform = make_env_uniform(EnvUniforms::NONE, "threers scene env uniforms");
        let no_env_uniform = make_env_uniform(EnvUniforms::NONE, "threers no env uniforms");
        let env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers env bg",
            &default_env_cube.view,
            &env_sampler,
            &dir_shadows.view,
            &shadow_sampler,
            &spot_shadows.view,
            &spot_shadow_sampler,
            &point_shadows.view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &env_uniform,
            &ltc.1,
        );
        // The trimmed custom-pipeline environment, where there is one. Rebuilt
        // alongside `env_bind_group` whenever the scene environment changes.
        let custom_env_bind_group = custom_env_bgl.as_ref().map(|bgl| {
            custom_env_bind_group!(
                device,
                bgl,
                "threers custom env bg",
                &default_env_cube.view,
                &env_sampler,
                &default_cube_uv_view,
                &env_uniform,
            )
        });

        // Dummy 1x1 depth textures used by `shadow_env_bind_group` so the
        // shadow passes can satisfy the pipeline layout without binding the
        // textures they're currently writing to (WebGPU forbids simultaneous
        // RenderAttachment + TextureBinding usage of the same resource).
        let dummy_depth = ShadowArray::new(&device, "threers dummy depth", 1, 1);
        let shadow_env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers shadow env bg (dummies)",
            &default_env_cube.view,
            &env_sampler,
            &dummy_depth.view,
            &shadow_sampler,
            &dummy_depth.view,
            &spot_shadow_sampler,
            &dummy_depth.view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &no_env_uniform,
            &ltc.1,
        );
        // Each draw's uniforms at an offset the device accepts for a binding.
        let mesh_stride = {
            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            (std::mem::size_of::<MeshUniforms>() as u64).div_ceil(align) * align
        };

        let mut renderer = Self {
            device,
            queue,
            pipeline_glass_depth,
            pipeline_glass_color,
            pipeline_glass_depth_f16,
            pipeline_glass_color_f16,
            pipeline_glass_depth_msaa,
            pipeline_glass_color_msaa,
            pipeline_glass_tint,
            pipeline_glass_tint_f16,
            pipeline_glass_tint_msaa,
            encoded_glass_msaa,
            pipeline_oit,
            pipeline_oit_resolve,
            pipeline_oit_resolve_f16,
            pipeline_oit_resolve_msaa,
            pipeline_oit_msaa,
            oit_msaa_targets: None,
            oit_msaa_targets_spare: None,
            depth_resolve,
            oit_resolve_bgl,
            oit_targets: None,
            ss_color: None,
            ss_glass_sampler,
            ss_depth: None,
            ss_back_depth: None,
            ss_layer: None,
            refraction_layers: 2,
            pipeline_ss_back_depth,
            pipeline_ss_blit,
            pipeline_ss_blit_f16,
            pipeline_ss_glass,
            pipeline_ss_glass_f16,
            pipeline_glass_shadow,
            ss_glass_shadow: None,
            ss_blit_bgl,
            taa_enabled: false,
            taa_frame: 0,
            taa_converge: false,
            taa_blit: true,
            taa_hist_format: None,
            jitter_override: None,
            taa_bind_cache: None,
            taa_prev_view_proj: [0.0; 16],
            mb_inv_vp: [0.0; 16],
            mb_prev_vp: [0.0; 16],
            mb_last_vp: [0.0; 16],
            taa_prev_cam: [0.0; 3],
            taa_history: None,
            taa_uniform,
            taa_bgl,
            taa_sampler,
            pipeline_taa,
            pipeline_taa_f16,
            pipeline_tri,
            pipeline_tri_alpha,
            pipeline_tri_nocull,
            pipeline_tri_sky,
            pipeline_tri_wire,
            pipeline_line,
            pipeline_point,
            pipeline_point_alpha,
            pipeline_wide_line,
            pipeline_wide_line_alpha,
            pipeline_sprite,
            pipeline_tri_msaa,
            pipeline_skinned_msaa,
            pipeline_instanced_msaa,
            pipeline_tri_alpha_msaa,
            pipeline_tri_nocull_msaa,
            pipeline_tri_sky_msaa,
            pipeline_tri_wire_msaa,
            pipeline_line_msaa,
            pipeline_point_msaa,
            pipeline_wide_line_msaa,
            msaa: 1,
            // Off by default: enabling tone mapping would silently re-grade
            // every existing scene and the parity renders along with them.
            tone_mapping: ToneMapping::None,
            specular_antialiasing: false,
            tone_map_pass: true,
            exposure: 1.0,
            msaa_targets: None,
            msaa_targets_spare: None,
            overlay_msaa: false,
            last_frame_msaa: false,
            fxaa_scratch: None,
            pipeline_tri_f16,
            pipeline_tri_alpha_f16,
            pipeline_tri_nocull_f16,
            pipeline_tri_sky_f16,
            pipeline_tri_wire_f16,
            pipeline_line_f16,
            pipeline_point_f16,
            pipeline_point_alpha_f16,
            pipeline_wide_line_f16,
            pipeline_wide_line_alpha_f16,
            pipeline_sprite_f16,
            pipeline_instanced_f16,
            pipeline_skinned_f16,
            pipeline_shadow,
            pipeline_shadow_spot,
            pipeline_shadow_point,
            pipeline_shadow_skinned,
            pipeline_shadow_spot_skinned,
            pipeline_shadow_point_skinned,
            pipeline_shadow_instanced,
            pipeline_shadow_spot_instanced,
            shadow_cut,
            pipeline_shadow_point_instanced,
            pipeline_instanced,
            pipeline_skinned,
            pipeline_instanced_wire,
            pipeline_instanced_wire_f16,
            pipeline_instanced_alpha,
            pipeline_instanced_alpha_f16,
            pipeline_skinned_alpha,
            pipeline_skinned_alpha_f16,
            encoded_blend_pipelines,
            pipeline_skinned_wire,
            pipeline_skinned_wire_f16,
            pipeline_postfx,
            pipeline_postfx_add,
            pipeline_postfx_outline_add,
            pipeline_postfx_multiply,
            pipeline_postfx_f16,
            pipeline_postfx_add_f16,
            postfx_bgl,
            postfx_sampler,
            postfx_nearest_sampler,
            postfx_noise_sampler,
            postfx_ext: Default::default(),
            ssao_kernel_buffer,
            ssao_noise_tex,
            ssao_noise_view,
            glitch_disp_tex,
            glitch_disp_view,
            rgba_upload_tex,
            rgba_upload_view,
            #[cfg(feature = "captions")]
            caption_pass: None,
            overlay_pass: None,
            mesh_skinned_bgl,
            sprite_quad,
            frame_bgl,
            mesh_bgl,
            tex_bgl,
            env_bgl,
            env_bgl_glass,
            ltc,
            custom_tex_bind_group,
            custom_tex_bgl,
            mesh_shader: shader.clone(),
            mesh_pipeline_layout: pipeline_layout.clone(),
            biased_pipelines: HashMap::new(),
            state_pipelines: state_pipes.into_inner(),
            msaa_color_unorm: None,
            encoded_scratch: None,
            encoded_copy: None,
            custom_mesh_bgl,
            custom_pipeline_layout,
            custom_ss_pipeline_layout,
            custom_pipelines: HashMap::new(),
            custom_encoded_pipelines: HashMap::new(),
            custom_msaa_pipelines: HashMap::new(),
            custom_failed: std::collections::HashSet::new(),
            depth_view,
            depth_texture,
            hw_depth_tex,
            depth_size: (width, height),
            geom_cache: HashMap::new(),
            frustum_culling: true,
            cull_stats: (0, 0),
            timer: None,
            shadow_timer: None,
            shadow_map: Default::default(),
            shadow_sig: None,
            shadow_pass_hashes: Vec::new(),
            shadow_stats: (0, 0),
            shadow_updates: 0,
            skin_attr_cache: HashMap::new(),
            morph_cache: HashMap::new(),
            instance_cache: HashMap::new(),
            instance_frame: 0,
            instance_stamps: 0,
            uv_cache: HashMap::new(),
            default_uv_sets,
            default_morph,
            wire_span_cache: HashMap::new(),
            tex_cache: HashMap::new(),
            mipgen: None,
            rt_view_cache: HashMap::new(),
            rt_linear_view_cache: HashMap::new(),
            rt_registry: HashMap::new(),
            rt_depth_view_cache: HashMap::new(),
            cube_rt_view_cache: HashMap::new(),
            env_max_mip: 0.0,
            cube_cache: HashMap::new(),
            env_bind_group,
            shadow_env_bind_group,
            custom_env_bgl,
            custom_env_bind_group,
            env_uniform,
            material_env_groups: HashMap::new(),
            pmrem_cache: HashMap::new(),
            env_sampler,
            default_env_cube,
            default_cube_uv_view,
            shadow_pass_buffer: None,
            depth_replay,
            depth_replay_pass: None,
            force_depth_replay: false,
            msaa_depth_out: None,
            depth_copy,
            shadow_pass_bgl,
            shadow_pass_stride,
            shadow_type: crate::lights::ShadowType::default(),
            env_cached_key: None,
            env_dirty: true,
            env_cached_rt_id: None,
            dir_shadows,
            spot_shadows,
            point_shadows,
            cube_pass: false,
            viewport_state: Default::default(),
            shadow_sampler,
            spot_shadow_sampler,
            point_shadow_sampler,
            frame_buffer,
            frame_bind_group,
            frame_bind_group_dummy,
            frame_buffer_for_bg,
            transmission_sampler,
            transmission: None,
            capturing_transmission: None,
            reflective_transparency: false,
            per_mesh: Vec::new(),
            mesh_arena: None,
            mesh_stride,
            mesh_bytes: Vec::new(),
            tex_bg_cache: HashMap::new(),
            main_shader: shader.clone(),
            map_variants: HashMap::new(),
            variant_bg_cache: HashMap::new(),
            sampler_linear_repeat,
            samplers,
            default_white_srgb,
            default_white_linear,
            default_data_tex_view,
            default_volume_view,
            sampler_volume,
            last_camera_planes: (0.1, 1000.0),
            last_camera_lens: (0.414, 1.0),
            default_normal,
            default_transparent,
            color_format,
        };

        // Screen-space ambient occlusion needs a sample kernel and a rotation
        // noise texture, and until now the only code that ever uploaded them
        // was `wasm.rs`. Every native caller got a zero kernel — so every
        // sample offset was the zero vector, every sample landed on the
        // fragment it started from, `delta` never exceeded `min_dist`, and the
        // effect returned a constant for every pixel whatever radius it was
        // given. That is why SSAO has been switched off by default: it was not
        // subtle, it was doing nothing.
        //
        // Defaults are installed here so the feature works out of the box; the
        // setters still override them.
        renderer.timer = GpuTimer::new(&renderer.device, &renderer.queue);
        renderer.shadow_timer = GpuTimer::new(&renderer.device, &renderer.queue);
        renderer.set_ssao_kernel(&default_ssao_kernel());
        renderer.set_ssao_noise(&default_ssao_noise());
        renderer
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

    /// three.js `renderer.clear()` for colour: fill `target_view` with `color`
    /// — the scissor rectangle when the scissor test is on, the whole target
    /// otherwise. `target_view` is taken to be drawing-buffer sized, as for
    /// [`render`](Self::render); `linear_framebuffer` means the same as there.
    pub fn clear(
        &mut self,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
        color: crate::math::Color,
        alpha: f32,
    ) {
        let size = self.depth_size;
        let format = if linear_framebuffer {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.color_format
        };
        self.clear_inner(target_view, size, format, color, alpha);
    }

    /// [`clear`](Self::clear) into an offscreen [`RenderTarget`](super::RenderTarget).
    pub fn clear_render_target(
        &mut self,
        target: &super::RenderTarget,
        color: crate::math::Color,
        alpha: f32,
    ) {
        self.clear_inner(
            &target.color_view,
            (target.width, target.height),
            target.format,
            color,
            alpha,
        );
    }

    fn clear_inner(
        &mut self,
        view: &wgpu::TextureView,
        (w, h): (u32, u32),
        format: wgpu::TextureFormat,
        color: crate::math::Color,
        alpha: f32,
    ) {
        use super::viewport::ClearMode;
        let Some(mode) = self.viewport_state.settings.clear_mode(w, h) else {
            return;
        };
        let value = super::clear_value(format, color, alpha);
        if let ClearMode::Rect(_) = mode {
            self.viewport_state
                .ensure_clear_pipeline(&self.device, format, None);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers clear"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if mode == ClearMode::Full {
                            wgpu::LoadOp::Clear(value)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let ClearMode::Rect(r) = mode {
                self.viewport_state.fill_rect(&mut pass, format, false, r, value);
            }
        }
        self.queue.submit(Some(encoder.finish()));
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

    /// Ensure the multisampled color+depth targets exist at `w×h` for the MSAA pass.
    /// The `Unorm` scratch target and the copy pipelines for the encoded-blend
    /// pass on a device without view-format reinterpretation (WebGL2).
    fn ensure_encoded_scratch(&mut self, w: u32, h: u32) {
        let unorm = self.color_format.remove_srgb_suffix();
        if !matches!(self.encoded_scratch, Some((cw, ch, _)) if cw == w && ch == h) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers encoded blend scratch"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.encoded_scratch = Some((w, h, tex.create_view(&Default::default())));
        }
        if self.encoded_copy.is_some() {
            return;
        }
        // `textureLoad` of the sRGB target decodes; `fs_in` encodes again, so
        // the `Unorm` scratch holds the target's bytes. `fs_out` reads the
        // scratch's bytes raw and decodes them, and the sRGB target encodes
        // on write: the bytes come back as they are. Both directions are
        // exact for 8-bit values.
        const COPY: &str = r#"
@group(0) @binding(0) var src : texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs_in(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let v = textureLoad(src, vec2<i32>(p.xy), 0);
    let c = clamp(v.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let e = select(pow(c, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055), c * 12.92, c <= vec3<f32>(0.0031308));
    return vec4<f32>(e, v.a);
}
@fragment
fn fs_out(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let v = textureLoad(src, vec2<i32>(p.xy), 0);
    let d = select(pow((v.rgb + 0.055) / 1.055, vec3<f32>(2.4)), v.rgb / 12.92, v.rgb <= vec3<f32>(0.04045));
    return vec4<f32>(d, v.a);
}
"#;
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers encoded blend copy"),
            source: wgpu::ShaderSource::Wgsl(COPY.into()),
        });
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers encoded blend copy bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers encoded blend copy layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry: &str, format: wgpu::TextureFormat| {
            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers encoded blend copy"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
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
        };
        let copy_in = pipe("fs_in", unorm);
        let copy_out = pipe("fs_out", self.color_format);
        self.encoded_copy = Some((bgl, copy_in, copy_out));
    }

    /// One fullscreen copy for the encoded-blend scratch (see
    /// [`ensure_encoded_scratch`](Self::ensure_encoded_scratch)).
    fn encoded_copy_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        out: bool,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let Some((bgl, copy_in, copy_out)) = &self.encoded_copy else {
            return;
        };
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers encoded blend copy bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers encoded blend copy pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(if out { copy_out } else { copy_in });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn ensure_msaa_targets(&mut self, w: u32, h: u32) {
        if !take_sized(&mut self.msaa_targets, &mut self.msaa_targets_spare, w, h) {
            return;
        }
        // The colour also viewable as its `Unorm` twin when it is sRGB, for
        // the encoded-blend pass.
        let unorm = self.color_format.remove_srgb_suffix();
        let reinterpret = unorm != self.color_format
            && super::caps::supports_view_formats(&self.device);
        let extra = [unorm];
        // The depth also sampled where the refraction pass resolves it.
        let sampled_depth = self.depth_resolve.is_some();
        let mk = |label, format, view_formats: &[wgpu::TextureFormat]| {
            let sampled = sampled_depth && format == DEPTH_FORMAT;
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.msaa,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: if sampled {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats,
            })
        };
        // wgpu keeps a texture alive as long as one of its views exists, so we can
        // store only the views.
        let color = mk(
            "threers msaa color",
            self.color_format,
            if reinterpret { &extra } else { &[] },
        );
        let cv = color.create_view(&Default::default());
        self.msaa_color_unorm = reinterpret.then(|| {
            color.create_view(&wgpu::TextureViewDescriptor {
                format: Some(unorm),
                ..Default::default()
            })
        });
        let dv = mk("threers msaa depth", DEPTH_FORMAT, &[]).create_view(&Default::default());
        self.msaa_targets = Some((w, h, cv, dv));
    }

    /// Apply a post-fx pass from a registered render-target id (wasm path).
    ///
    /// Same as [`Self::apply_postfx`] but resolves `input_rt_id` internally so
    /// the caller does not hold a borrow against `&mut self`. When `additive`
    /// is true, output is blended onto existing framebuffer pixels.
    ///
    /// `kind` selects the WGSL branch in [`super::shader::POSTFX_SHADER`].
    /// `time` carries effect-specific data (e.g. glitch RNG seed). `params2` /
    /// `params3` are four-vectors whose meaning depends on `kind`.
    ///
    /// Every kind keeps the picture the right way up, into a render target
    /// (`output_rt_id != 0`) and onto the canvas (`0`) alike, so passes chain
    /// in any number without compensating flips.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx_by_id(
        &mut self,
        input_rt_id: u32,
        output_rt_id: u32,
        output_view: &wgpu::TextureView,
        depth_rt_id: u32,
        normal_rt_id: u32,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
    ) {
        let output_format = if output_rt_id != 0 {
            self.rt_registry
                .get(&output_rt_id)
                .map(|rt| rt.format)
                .unwrap_or(self.color_format)
        } else {
            self.color_format
        };
        let input_view = if let Some(v) = self.rt_view_cache.remove(&input_rt_id) {
            v
        } else if let Some(rt) = self.rt_registry.get(&input_rt_id) {
            rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers postfx input fallback"),
                format: Some(rt_sample_format(rt.format)),
                ..Default::default()
            })
        } else {
            return;
        };
        let depth_owned = if depth_rt_id != 0 && depth_rt_id != input_rt_id {
            self.rt_view_cache.remove(&depth_rt_id).or_else(|| {
                self.rt_registry.get(&depth_rt_id).map(|rt| {
                    rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("threers postfx depth fallback"),
                        format: Some(rt_sample_format(rt.format)),
                        ..Default::default()
                    })
                })
            })
        } else {
            None
        };
        let normal_owned = if normal_rt_id != 0 {
            self.rt_view_cache.remove(&normal_rt_id)
        } else {
            None
        };
        let depth_ref = depth_owned.as_ref().unwrap_or(&input_view);
        let input_is_f16 = self
            .rt_registry
            .get(&input_rt_id)
            .map(|rt| rt.format == wgpu::TextureFormat::Rgba16Float)
            .unwrap_or(false);
        // Kind 0 copies stored RT texels 1:1. Default: linear (non-sRGB-decode)
        // view so textureLoad returns the gamma-encoded bytes the main shader
        // wrote (matches direct-to-canvas). params2.y > 0.5 selects the legacy
        // sRGB-decode view for OutlinePass scene copy (three.js darkens there).
        let kind0_legacy = kind == 0 && params2[1] > 0.5;
        let output_is_f16 = output_format == wgpu::TextureFormat::Rgba16Float;
        let kind0_byte_copy = kind == 0 && !kind0_legacy && !input_is_f16 && !output_is_f16;
        let use_linear_input = if kind == 0 {
            input_is_f16 && !kind0_legacy
        } else {
            (matches!(kind, 13 | 15) && !kind0_legacy)
                || input_is_f16
                || matches!(kind, 16 | 18 | 19)
        };
        let use_linear_depth = matches!(kind, 11 | 12 | 14);
        let input_linear = if use_linear_input {
            self.rt_registry.get(&input_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear input"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let input_raw = if kind0_byte_copy {
            self.rt_registry.get(&input_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx raw input"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let depth_linear = if use_linear_depth && depth_rt_id != 0 {
            self.rt_registry.get(&depth_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear depth"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let normal_linear = if use_linear_depth {
            self.rt_registry.get(&normal_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear normal"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let input_bind = input_raw
            .as_ref()
            .or(input_linear.as_ref())
            .unwrap_or(&input_view);
        let depth_bind = depth_linear.as_ref().unwrap_or(depth_ref);
        let mut params2 = params2;
        if kind0_byte_copy {
            params2[2] = 1.0;
        }
        // SSAO (11) and bokeh (26) read the hardware depth of the prepass
        // target, which the caller names as both depth and normal.
        let use_hw_depth =
            matches!(kind, 11 | 26) && depth_rt_id != 0 && depth_rt_id == normal_rt_id;
        let hw_depth_local = if use_hw_depth {
            self.rt_depth_view_cache.remove(&normal_rt_id)
        } else {
            None
        };
        let hw_depth_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers postfx hw depth fallback"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let hw_depth_bind = hw_depth_local.as_ref().unwrap_or(&hw_depth_fallback);
        // A "raw" view above is the target's own format: without view formats
        // (WebGL2) an sRGB target cannot be viewed as anything else, so the
        // view decodes and the shader has to re-encode what it reads.
        let srgb_rt = |id: u32| {
            self.rt_registry
                .get(&id)
                .is_some_and(|rt| rt.format.is_srgb())
        };
        let mut raw_srgb = 0u32;
        if (input_raw.is_some() || input_linear.is_some()) && srgb_rt(input_rt_id) {
            raw_srgb |= 2;
        }
        if depth_linear.is_some() && srgb_rt(depth_rt_id) {
            raw_srgb |= 4;
        }
        if normal_linear.is_some() && srgb_rt(normal_rt_id) {
            raw_srgb |= 8;
        }
        match normal_owned {
            Some(n) => {
                let normal_bind = normal_linear.as_ref().unwrap_or(&n);
                self.apply_postfx_views(
                    input_bind,
                    output_view,
                    depth_bind,
                    normal_bind,
                    hw_depth_bind,
                    kind,
                    time,
                    params2,
                    params3,
                    width,
                    height,
                    additive,
                    camera,
                    use_hw_depth,
                    output_format,
                    raw_srgb,
                );
                self.rt_view_cache.insert(normal_rt_id, n);
                self.tex_bg_cache.clear();
                self.variant_bg_cache.clear();
            }
            None => {
                let fallback_normal = self
                    .default_normal
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let normal_bind = normal_linear.as_ref().unwrap_or(&fallback_normal);
                self.apply_postfx_views(
                    input_bind,
                    output_view,
                    depth_bind,
                    normal_bind,
                    hw_depth_bind,
                    kind,
                    time,
                    params2,
                    params3,
                    width,
                    height,
                    additive,
                    camera,
                    use_hw_depth,
                    output_format,
                    raw_srgb,
                );
            }
        }
        if let Some(v) = hw_depth_local {
            self.rt_depth_view_cache.insert(normal_rt_id, v);
        }
        self.rt_view_cache.insert(input_rt_id, input_view);
        self.tex_bg_cache.clear();
        self.variant_bg_cache.clear();
        if let Some(d) = depth_owned {
            self.rt_view_cache.insert(depth_rt_id, d);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
    }

    /// Apply a post-fx pass: read from `input_view`, write to `output_view`.
    /// Defaults to REPLACE blend / clear on load. Use `apply_postfx_ex` for
    /// additive blend (bloom composite) or load-preserve behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
    ) {
        let fallback_normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.apply_postfx_ex(
            input_view,
            output_view,
            input_view,
            &fallback_normal,
            &hw_fallback,
            kind,
            time,
            params2,
            params3,
            width,
            height,
            false,
            PostFxCamera::default(),
            false,
            self.color_format,
        );
    }

    /// Extended post-fx pass with `additive` blend toggle.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx_ex(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        hw_depth_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
        use_hw_depth: bool,
        output_format: wgpu::TextureFormat,
    ) {
        self.apply_postfx_views(
            input_view,
            output_view,
            depth_view,
            normal_view,
            hw_depth_view,
            kind,
            time,
            params2,
            params3,
            width,
            height,
            additive,
            camera,
            use_hw_depth,
            output_format,
            0,
        );
    }

    /// [`Self::apply_postfx_ex`], told which bound views were meant to show
    /// the stored bytes but decode them: `raw_srgb` holds `POSTFX_SHADER`'s
    /// `postfx_flag` bits 2 (input), 4 (depth) and 8 (normal). Bit 1 — the
    /// output is an sRGB format — is worked out here from `output_format`.
    ///
    /// Each branch of the shader is written for a plain `Unorm` target: what
    /// it returns is the byte stored. An sRGB target encodes on write, so
    /// without bit 1 every computed pass was encoded once more on WebGL2 —
    /// where every 8-bit target and the canvas are sRGB — than on WebGPU.
    #[allow(clippy::too_many_arguments)]
    fn apply_postfx_views(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        hw_depth_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
        use_hw_depth: bool,
        output_format: wgpu::TextureFormat,
        raw_srgb: u32,
    ) {
        let linear_output = output_format == wgpu::TextureFormat::Rgba16Float;
        // Bit 16: the hardware depth is reversed (`set_reversed_z`), and
        // `sample_postfx_depth` hands back `1 - d`.
        let srgb_bits = raw_srgb
            | u32::from(output_format.is_srgb())
            | if self.state_pipelines.reversed() { 16 } else { 0 };
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PostFxUniforms {
            params: [f32; 4],
            params2: [f32; 4],
            params3: [f32; 4],
            resolution: [f32; 4],
            ssao0: [f32; 4],
            inv_proj: [f32; 16],
            proj: [f32; 16],
        }
        let uniforms = PostFxUniforms {
            params: [
                kind as f32,
                time,
                if matches!(kind, 5 | 31) {
                    params3[3]
                } else {
                    0.0
                },
                if use_hw_depth { 1.0 } else { 0.0 },
            ],
            params2,
            params3,
            resolution: [
                width as f32,
                height as f32,
                if linear_output { 1.0 } else { 0.0 },
                srgb_bits as f32,
            ],
            ssao0: [
                camera.near,
                camera.far,
                camera.kernel_radius,
                camera.kernel_size as f32,
            ],
            inv_proj: camera.inv_proj,
            proj: camera.proj,
        };
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers postfx uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers postfx bg"),
            layout: &self.postfx_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
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
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(
                        if matches!(kind, 5 | 31) {
                            &self.postfx_sampler
                        } else {
                            &self.postfx_nearest_sampler
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hw_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.ssao_kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(if kind == 5 || kind == 31 {
                        &self.glitch_disp_view
                    } else {
                        &self.ssao_noise_view
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.postfx_noise_sampler),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers postfx encoder"),
            });
        {
            let multiply = kind == 15;
            let load_op = if additive || multiply {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers postfx pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let output_is_f16 = output_format == wgpu::TextureFormat::Rgba16Float;
            let pipe = if multiply {
                &self.pipeline_postfx_multiply
            } else if additive {
                if kind == 17 {
                    &self.pipeline_postfx_outline_add
                } else if output_is_f16 {
                    &self.pipeline_postfx_add_f16
                } else {
                    &self.pipeline_postfx_add
                }
            } else if output_is_f16 {
                &self.pipeline_postfx_f16
            } else {
                &self.pipeline_postfx
            };
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Upload SSAO hemisphere kernel samples (32 vec3 values, padded to vec4).
    pub fn set_ssao_kernel(&mut self, kernel: &[f32]) {
        let mut data = [0f32; 128];
        for i in 0..32 {
            let base = i * 3;
            if base + 2 < kernel.len() {
                data[i * 4] = kernel[base];
                data[i * 4 + 1] = kernel[base + 1];
                data[i * 4 + 2] = kernel[base + 2];
            }
        }
        self.queue
            .write_buffer(&self.ssao_kernel_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Blit tightly-packed RGBA8 bytes (row-major, top-first) to an output view.
    pub fn blit_rgba8(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        output_view: &wgpu::TextureView,
        output_format: wgpu::TextureFormat,
    ) {
        let w = width.max(1);
        let h = height.max(1);
        let need = (w * h * 4) as usize;
        if data.len() < need {
            return;
        }
        // RGBA8 bytes from a canvas or an image are sRGB-encoded and are to
        // reach the screen as they came. They are uploaded as plain `Unorm`,
        // so the copy reads the bytes themselves; onto an sRGB target the
        // post-fx pass decodes its output (`postfx_flag` bit 1) and the
        // target's encode gives the same bytes back.
        let upload_format = wgpu::TextureFormat::Rgba8Unorm;
        if self.rgba_upload_tex.width() != w
            || self.rgba_upload_tex.height() != h
            || self.rgba_upload_tex.format() != upload_format
        {
            self.rgba_upload_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers rgba upload"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: upload_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.rgba_upload_view = self
                .rgba_upload_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        const ALIGN: u32 = 256;
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(ALIGN) * ALIGN;
        let mut upload = vec![0u8; (padded * h) as usize];
        for row in 0..h {
            let src = (row * w * 4) as usize;
            let dst = (row * padded) as usize;
            upload[dst..dst + (w * 4) as usize].copy_from_slice(&data[src..src + (w * 4) as usize]);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.rgba_upload_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let upload_view = self
            .rgba_upload_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        let fallback_normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.apply_postfx_ex(
            &upload_view,
            output_view,
            &upload_view,
            &fallback_normal,
            &hw_fallback,
            0,
            0.0,
            // A raw texel copy (`params2.z`): the bytes are already what the
            // screen should show, so no gamma is applied on the way out.
            [1.0, 0.0, 1.0, 0.0],
            [0.0; 4],
            w,
            h,
            false,
            PostFxCamera::default(),
            false,
            output_format,
        );
    }

    /// Drop every cached GPU mesh, freeing their buffers.
    ///
    /// The cache is keyed by the address of the geometry's `Arc`, which is
    /// stable only while that `Arc` is alive. Code that feeds the renderer a
    /// *fresh* geometry every frame — a CAD model re-evaluated per frame, say —
    /// must call this between frames: without it the cache grows without bound,
    /// and an allocator that reuses a freed address could serve the old mesh
    /// for the new geometry.
    ///
    /// Scenes that keep their geometry alive (the usual case) never need this.
    pub fn clear_geometry_cache(&mut self) {
        self.geom_cache.clear();
        self.skin_attr_cache.clear();
        self.morph_cache.clear();
        self.wire_span_cache.clear();
    }

    /// How many GPU meshes are currently cached.
    pub fn cached_geometry_count(&self) -> usize {
        self.geom_cache.len()
    }

    /// Drop every cached GPU texture.
    ///
    /// The texture cache has exactly the same shape, and the same hazard, as
    /// the geometry one: it is keyed by the address of the `Texture`'s `Arc`,
    /// so code that builds fresh textures every frame must clear it or risk an
    /// allocator handing a new texture the address of a freed one and getting
    /// the previous upload back.
    /// Upload a texture now, without drawing anything with it.
    ///
    /// Textures normally upload lazily, on the first draw that binds them. That
    /// is the wrong moment when the pixels are large and numerous: each has to
    /// be held in system memory until something draws it, so the peak is the
    /// sum of all of them. Preloading one at a time and swapping in
    /// [`Texture::handle`] keeps the peak at one.
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Texture, TextureFormat};
    /// # use std::sync::Arc;
    /// # let mut hr = HeadlessRenderer::builder().size(64, 64).build().unwrap();
    /// # let load = |_: &str| Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![0; 4]);
    /// let full = Arc::new(load("tile.tex"));
    /// hr.renderer().preload_texture(&full);
    /// let light = Arc::new(full.handle()); // same id, no pixels
    /// drop(full);                          // system memory released
    /// ```
    pub fn preload_texture(&mut self, tex: &std::sync::Arc<crate::textures::Texture>) {
        self.ensure_slot_uploaded(&Some(tex.clone()));
    }

    pub fn clear_texture_cache(&mut self) {
        self.tex_cache.clear();
        self.cube_cache.clear();
    }

    /// How many GPU textures are currently cached.
    pub fn cached_texture_count(&self) -> usize {
        self.tex_cache.len()
    }

    /// Blend a straight-alpha RGBA8 overlay over an already-rendered target.
    ///
    /// Unlike [`blit_rgba8`](Self::blit_rgba8), which replaces the target, this
    /// loads the existing contents and composites `data` on top — the 3D image
    /// stays visible wherever the overlay is transparent. `data` must be
    /// `width * height * 4` bytes, row-major from the top-left.
    ///
    /// Draw it after [`render`](Self::render) and before presenting the frame.
    pub fn draw_overlay_rgba8(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        self.composite_overlay(data, width, height, None, target, format);
    }

    /// Blend an [`Overlay`](crate::overlay::Overlay) over an already-rendered
    /// target, uploading only the rows that changed.
    ///
    /// Prefer this to [`draw_overlay_rgba8`](Self::draw_overlay_rgba8) when the
    /// pixels come from an `Overlay`. A full-frame upload is most of what
    /// compositing one costs — at 1920x1080 the upload alone measures 728 µs
    /// against ~124 µs for the quad and the submit together — and a read-out in
    /// the corner changes 95 rows out of 1080. Measured over a frame that
    /// clears the overlay, draws a two-line panel and composites it:
    /// **1.1 ms against 223 µs, 4.9x**.
    ///
    /// Taking the `Overlay` rather than its bytes is what makes that safe. The
    /// band to upload and the acknowledgement that it happened are two halves
    /// of one step, and a caller doing them by hand could skip a composite and
    /// leave the texture holding a frame nobody remembers.
    ///
    /// Draw it after [`render`](Self::render) and before presenting. Use one
    /// `Overlay` per `Renderer`: the dirty band describes what *this*
    /// renderer's texture is missing.
    pub fn draw_overlay(
        &mut self,
        overlay: &mut crate::overlay::Overlay,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        let (width, height) = overlay.size();
        // An empty band uploads nothing and still composites, which is what a
        // frame wants when the overlay has not changed: the image underneath it
        // is new even when the overlay is not.
        let rows = overlay.dirty_rows().unwrap_or((0, 0));
        self.composite_overlay(overlay.pixels(), width, height, Some(rows), target, format);
        overlay.mark_uploaded();
    }

    fn composite_overlay(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        rows: Option<(u32, u32)>,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        let pass = self
            .overlay_pass
            .get_or_insert_with(|| super::caption_pass::CaptionPass::new(&self.device));
        pass.draw(
            &self.device,
            &self.queue,
            data,
            width,
            height,
            rows,
            target,
            format,
        );
    }

    /// Draw the caption showing at `time` seconds over an already-rendered target.
    ///
    /// Sizes the overlay to `target_width × target_height`, rasterizes the
    /// active cue only when it changes, and blends it in. Cheap enough to call
    /// every frame — a frame with no cue change costs one texture upload, and a
    /// frame with no cue at all costs nothing.
    ///
    /// ```no_run
    /// # use threers::{Renderer, Scene, PerspectiveCamera, captions::{CaptionOverlay, CaptionTrack}};
    /// # fn demo(renderer: &mut Renderer, scene: &mut Scene, camera: &PerspectiveCamera,
    /// #         view: &wgpu::TextureView, format: wgpu::TextureFormat, time: f64) {
    /// let mut overlay = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 3.0, "Hello"))
    ///     .auto_scale(true);
    /// renderer.render(scene, camera, view, false);
    /// renderer.draw_caption_overlay(&mut overlay, time, 1280, 720, view, format);
    /// # }
    /// ```
    #[cfg(feature = "captions")]
    pub fn draw_caption_overlay(
        &mut self,
        overlay: &mut crate::captions::CaptionOverlay,
        time: f64,
        target_width: u32,
        target_height: u32,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        overlay.set_size(target_width, target_height);
        overlay.update(time);
        if !overlay.is_visible() {
            return;
        }
        let (w, h) = overlay.size();
        // Borrow the buffer out of the overlay for the upload; the pass only
        // reads it, but the borrow checker needs the copy-free split.
        let pass = self
            .caption_pass
            .get_or_insert_with(|| super::caption_pass::CaptionPass::new(&self.device));
        pass.draw(
            &self.device,
            &self.queue,
            overlay.rgba(),
            w,
            h,
            None,
            target,
            format,
        );
    }

    /// Upload GlitchPass's `size × size` displacement heightmap (R32Float, read
    /// nearest, as three.js's `DataTexture` defaults to).
    pub fn set_glitch_disp(&mut self, data: &[f32], size: u32) {
        let sz = size.max(1);
        if data.len() < (sz * sz) as usize {
            return;
        }
        if self.glitch_disp_tex.width() != sz {
            self.glitch_disp_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers glitch disp"),
                size: wgpu::Extent3d {
                    width: sz,
                    height: sz,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.glitch_disp_view = self
                .glitch_disp_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        let data = &data[..(sz * sz) as usize];
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.glitch_disp_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sz * 4),
                rows_per_image: Some(sz),
            },
            wgpu::Extent3d {
                width: sz,
                height: sz,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload 4×4 SSAO rotation noise (R channel, repeat-wrapped).
    pub fn set_ssao_noise(&mut self, noise: &[f32]) {
        let mut texels = [0f32; 16];
        for (dst, src) in texels.iter_mut().zip(noise.iter()) {
            *dst = *src;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.ssao_noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Render `scene` from `camera` into one face of a cube render target.
    /// Used by `CubeCamera.update()` which calls this 6 times per frame.
    /// Reallocate the directional shadow map if a caster asks for a different
    /// size. Powers of two only, and bounded: this is a render target, and a
    /// caller who types 65536 should get the largest sane map rather than an
    /// allocation failure three frames later.
    /// Size the shadow arrays for this frame's casters: a layer per
    /// directional and spot light, six per point light, at the size the
    /// directional lights ask for (`ShadowSettings::map_size`, the largest)
    /// and [`SPOT_SHADOW_SIZE`] for spots, shrunk when several share a kind
    /// ([`shadow_layer_size`]). Reallocated only to grow or resize.
    fn ensure_shadow_arrays(&mut self, dir_size: u32, n_dir: usize, n_spot: usize, n_point: usize) {
        let mut changed = false;
        let mut fit = |array: &mut ShadowArray, label: &str, size: u32, layers: u32| {
            if array.size != size || array.layers < layers {
                *array = ShadowArray::new(&self.device, label, size, layers);
                changed = true;
            }
        };
        if n_dir > 0 {
            fit(&mut self.dir_shadows, "threers shadow depth", shadow_layer_size(dir_size, n_dir), n_dir as u32);
        }
        if n_spot > 0 {
            fit(&mut self.spot_shadows, "threers spot shadow depth", shadow_layer_size(SPOT_SHADOW_SIZE, n_spot), n_spot as u32);
        }
        if n_point > 0 {
            fit(&mut self.point_shadows, "threers point shadow depth", POINT_SHADOW_SIZE, 6 * n_point as u32);
        }
        if changed {
            // The environment bind group holds the old views. Invalidating the
            // cache key is what makes the next frame rebuild it; without this
            // the shader would keep sampling a texture nothing renders into
            // any more.
            self.env_cached_key = None;
            self.env_cached_rt_id = None;
            self.env_dirty = true;
        }
    }

    /// Room for `n` shadow passes' uniforms (`ShadowPassGpu`), one aligned
    /// slot each, and the group 0 the shadow pipelines read them through at
    /// a dynamic offset — beside the frame uniforms, which the passes' vertex
    /// stages may read too.
    ///
    /// One buffer for every pass, written once a frame, rather than a copy
    /// of the whole frame uniforms per pass: the passes are all encoded
    /// before anything runs and `queue.write_buffer` lands before any
    /// submitted command, so each pass needs its own slot either way.
    fn ensure_shadow_pass_buffer(&mut self, n: usize) {
        let need = self.shadow_pass_stride * n.max(1) as u64;
        if self.shadow_pass_buffer.as_ref().is_some_and(|(b, _)| b.size() >= need) {
            return;
        }
        let size = need.max(self.shadow_pass_stride * 8);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers shadow pass uniforms"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers shadow pass bg"),
            layout: &self.shadow_pass_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.frame_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<ShadowPassGpu>() as u64),
                    }),
                },
            ],
        });
        self.shadow_pass_buffer = Some((buffer, bg));
    }

    pub fn render_to_cube_face(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::CubeRenderTarget,
        face: usize,
    ) {
        // Tell the frame passes they are drawing a probe, not the picture.
        let was_cube = std::mem::replace(&mut self.cube_pass, true);
        let (orig_w, orig_h) = self.depth_size;
        let orig_depth = std::mem::replace(
            &mut self.depth_view,
            target
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        self.depth_size = (target.side, target.side);
        let face_view = &target.face_views[face % 6];
        self.render(scene, camera, face_view, false);
        self.depth_view = orig_depth;
        self.depth_size = (orig_w, orig_h);
        self.cube_pass = was_cube;
    }

    /// Render `scene` from `camera` into an offscreen [`RenderTarget`](crate::RenderTarget). Uses
    /// the target's own depth buffer (required even when RT size matches the canvas).
    pub fn render_to(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::RenderTarget,
    ) {
        let (orig_w, orig_h) = self.depth_size;
        let orig_depth = std::mem::replace(&mut self.depth_view, target.depth_view_clone());
        self.depth_size = (target.width, target.height);
        let linear_fb = target.format == wgpu::TextureFormat::Rgba16Float;
        let tone_map = std::mem::replace(&mut self.tone_map_pass, target.presents());
        // Resolving needs the multisampled colour in the target's format, and
        // the renderer's multisampled targets are in its own.
        if target.samples() > 1 && target.format == self.color_format {
            // three.js `samples`: drawn multisampled at the target's size,
            // colour resolved into it and depth resolved (or, on WebGL2,
            // redrawn) into its depth, which the glass and post-processing
            // read as they would a single-sampled target's. The renderer's own
            // MSAA setting is the canvas's and is left as it was.
            let msaa = std::mem::replace(&mut self.msaa, MSAA_SAMPLES);
            self.msaa_depth_out = Some(target.depth_texture.clone());
            self.render_inner(
                scene,
                camera,
                &target.color_view,
                linear_fb,
                None,
                Some((&target.depth_texture, &target.color_texture)),
            );
            self.msaa_depth_out = None;
            self.msaa = msaa;
        } else {
            // The target owns the live depth this pass — pass it as the copy
            // source for the screen-space glass ray march.
            self.render_inner(
                scene,
                camera,
                &target.color_view,
                linear_fb,
                Some(&target.depth_texture),
                Some((&target.depth_texture, &target.color_texture)),
            );
        }
        self.tone_map_pass = tone_map;
        self.depth_view = orig_depth;
        self.depth_size = (orig_w, orig_h);
    }

    /// Draw into a [`RenderTarget`](super::RenderTarget) the surface-style way —
    /// the path MSAA needs — while still handing the temporal pass the textures
    /// it reads.
    ///
    /// [`render`](Self::render) passes `None` for both, which silently turns TAA
    /// off: it is gated on having a colour source, and that path has none to
    /// give. Multisampling and temporal accumulation are complementary — one
    /// resolves coverage, the other resolves shading — and a caller that asked
    /// for both got only the first, with nothing said about it. Measured on a
    /// phone's side rail, thirty-two accumulated samples took the banding from
    /// 0.99 to 0.28 with MSAA off, and did exactly nothing with MSAA on.
    ///
    /// The depth handed over is the target's, which the multisampled path does
    /// not write. That is why this is separate from [`render_to`](Self::render_to)
    /// rather than folded into it: the temporal pass binds depth but only reads
    /// it when reprojecting, which [`set_taa_converge`](Self::set_taa_converge)
    /// does not do, and the screen-space glass march — the other consumer of a
    /// live depth — is deliberately not given it here.
    pub fn render_to_multisampled(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::RenderTarget,
        linear_framebuffer: bool,
    ) {
        let tone_map = std::mem::replace(&mut self.tone_map_pass, target.presents());
        self.last_frame_msaa = false;
        self.render_inner(
            scene,
            camera,
            &target.color_view,
            linear_framebuffer,
            None,
            Some((&target.depth_texture, &target.color_texture)),
        );
        self.tone_map_pass = tone_map;
        // Antialiasing was asked for and the frame could not multisample
        // (`Oit` or `Refract` glass in view): FXAA it rather than hand back
        // hard edges. Temporal AA, when on, has already done the job.
        if self.msaa > 1 && !self.last_frame_msaa && !self.taa_enabled {
            self.fxaa_fallback(target);
        }
    }

    /// FXAA over `target`, read from a copy of it.
    ///
    /// On the stored bytes, as three.js runs FXAA over its 8-bit canvas: an
    /// sRGB copy decodes as it is read, so the pass is told (`raw_srgb` bit 2)
    /// and re-encodes, and the sRGB target encodes the result back to the
    /// same bytes. A flat area comes back byte for byte.
    fn fxaa_fallback(&mut self, target: &super::RenderTarget) {
        let tex = &target.color_texture;
        let (w, h, format) = (tex.width(), tex.height(), tex.format());
        if format == wgpu::TextureFormat::Rgba16Float {
            // A linear target is a stage on the way somewhere else; whatever
            // presents it can antialias the result.
            return;
        }
        let stale = self
            .fxaa_scratch
            .as_ref()
            .is_none_or(|(t, _)| t.width() != w || t.height() != h || t.format() != format);
        if stale {
            let t = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers fxaa fallback source"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
            self.fxaa_scratch = Some((t, v));
        }
        let (scratch, view) = self.fxaa_scratch.as_ref().unwrap();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers fxaa fallback copy"),
        });
        encoder.copy_texture_to_texture(
            tex.as_image_copy(),
            scratch.as_image_copy(),
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let view = view.clone();
        let normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_depth = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let raw_input = if format.is_srgb() { 2 } else { 0 };
        self.apply_postfx_views(
            &view,
            &target.color_view,
            &view,
            &normal,
            &hw_depth,
            1,
            0.0,
            [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0],
            [0.0; 4],
            w,
            h,
            false,
            PostFxCamera::default(),
            false,
            format,
            raw_input,
        );
    }

    /// Draw `scene` from `camera` into `target_view`.
    /// `linear_framebuffer`: when true, skip sRGB encode (HalfFloat / postfx RT).
    pub fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
    ) {
        self.render_inner(scene, camera, target_view, linear_framebuffer, None, None);
    }

    /// Body of [`render`]. `depth_src` is the depth texture backing the current
    /// depth attachment (the external target's, via `render_to`) — the copy
    /// source for the screen-space glass `ss_depth`; `None` uses the renderer's
    /// own depth (the direct-`render`/surface path).
    fn render_inner(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
        depth_src: Option<&wgpu::Texture>,
        // `(depth, colour)` for the temporal pass. Separate from `depth_src`,
        // which is the *live* depth for the screen-space glass march — the
        // multisampled path has one and not the other.
        taa_src: Option<(&wgpu::Texture, &wgpu::Texture)>,
    ) {
        scene.update_world();
        if self.capturing_transmission.is_none() && !self.cube_pass {
            self.capture_transmission(scene, camera);
        }

        // Viewport / scissor for this render. A cube face is always the whole
        // face. `None` means the viewport or scissor misses the target and
        // there is nothing to draw.
        let vp_plan = if self.cube_pass {
            super::viewport::ViewportPlan::full(self.depth_size.0, self.depth_size.1)
        } else {
            match self.viewport_state.settings.plan(self.depth_size.0, self.depth_size.1) {
                Some(p) => p,
                None => return,
            }
        };

        // Whether the shader writes linear light and lets something else encode
        // it. Two different reasons to: the target stores linear (a float
        // framebuffer, which is what `linear_framebuffer` says), or the target's
        // format is sRGB and the hardware encodes on write.
        //
        // This is *not* the same question as `linear_framebuffer`, which every
        // other use of that flag below is really asking — "is this a float
        // target, so pick the f16 pipeline". Conflating the two is what made an
        // sRGB target double-encode. Picking the format off `self.color_format`
        // rather than the target view is sound because every non-float target
        // the renderer draws into is the one its pipelines were built against.
        let shader_outputs_linear =
            linear_framebuffer || !super::shader_applies_transfer(self.color_format);

        // TAA active only when we have a sampleable color+depth target.
        // NOT during a probe. A cube face is 48 pixels and is not the picture:
        // running temporal AA over it reprojects the frame's history through
        // the probe's camera and writes the result back, so the next real frame
        // blends against six tiny views of the scene. On screen that reads as
        // the whole vehicle going half transparent — which is what it looked
        // like, and why the cause was not obvious from the symptom.
        let taa_active =
            self.taa_enabled && !self.cube_pass && taa_src.is_some() && vp_plan.legacy;

        let view_m = camera.view_matrix();
        let mut proj_m = camera.projection_matrix();
        if let Some(fix) = &vp_plan.projection_fix {
            proj_m = fix.multiply(&proj_m);
        }
        // Un-jittered view-projection is kept for TAA reprojection.
        let unjittered = proj_m.multiply(&view_m);
        if taa_active || self.jitter_override.is_some() {
            // Sub-pixel Halton jitter, half a pixel either way.
            //
            // Deliberately a box over the pixel and not a weighted
            // reconstruction: a pixel *is* a box, so weighting the samples by
            // a tent and widening the jitter to match was tried and measures
            // worse — the synthetic edge went from 0.0216 px of deviation to
            // 0.0318, because spreading the same samples over four times the
            // area undersamples it faster than the filter narrows it.
            let (ox, oy) = self.jitter_override.unwrap_or((
                halton(self.taa_frame + 1, 2) - 0.5,
                halton(self.taa_frame + 1, 3) - 0.5,
            ));
            // Per viewport pixel — the whole target unless a viewport is set.
            let jx = ox * 2.0 / vp_plan.viewport.width as f32;
            let jy = oy * 2.0 / vp_plan.viewport.height as f32;
            // Which element carries the offset depends on the projection, and
            // getting it wrong is not subtle. A perspective matrix divides by
            // `-z`, so a shift in the *z* column arrives in NDC as a constant:
            // `x_clip += jx·z` over `w = -z` is `-jx` at every depth. An
            // orthographic matrix does not divide — `w` is 1 — so that same
            // shift lands as `jx·z`, which is a shear that grows with distance
            // from the camera, not a sub-pixel offset. With the camera ten
            // units back it spread a hard edge over nine pixels. Orthographic
            // wants the translation column, where the offset is already in NDC.
            //
            // `elements[11]` is `-1` for perspective and `0` for orthographic,
            // which is the projection asked about rather than the camera type
            // guessed at.
            if proj_m.elements[11].abs() > 0.5 {
                proj_m.elements[8] += jx;
                proj_m.elements[9] += jy;
            } else {
                // Negated so the jitter means the same direction either way.
                proj_m.elements[12] -= jx;
                proj_m.elements[13] -= jy;
            }
        }
        // Reversed depth flips only what rasterises: TAA, motion blur and
        // culling keep the standard matrices, and every reader of the depth
        // buffer converts it back (`1 - d`) before using them.
        let reversed = self.state_pipelines.reversed();
        if reversed {
            proj_m = reverse_depth(&proj_m);
        }
        let view_proj = proj_m.multiply(&view_m).elements;
        let taa_inv_vp = unjittered.invert().elements;
        let taa_unjittered_vp = unjittered.elements;

        // tone_mapping_exposure[2] is the "IBL on" flag — the PBR shader's
        // env-map branch is gated on it. Enable whenever the scene has any
        // environment source (CPU cube texture or cube render target).
        let has_env = scene.environment.is_some() || scene.environment_cube_rt.is_some();
        let (cam_near, cam_far) = camera.near_far();
        // Remembered for the post passes. Depth linearisation needs these and
        // they are a property of the frame, not of the effect — making every
        // caller pass them again is an invitation to pass the wrong ones, and
        // the failure is silent: the occlusion or the blur is simply computed
        // at the wrong distances and still looks like an effect.
        self.last_camera_planes = (cam_near, cam_far);
        // The lens, from the projection itself: for a perspective matrix
        // `m[5]` is `1 / tan(fov/2)` and `m[0]` is that over the aspect.
        let proj = camera.projection_matrix();
        let (m0, m5) = (proj.elements[0], proj.elements[5]);
        if m5.abs() > 1e-9 && m0.abs() > 1e-9 {
            self.last_camera_lens = (1.0 / m5, m5 / m0);
        }
        let mut frame_u = FrameUniforms {
            view: view_m.elements,
            projection: proj_m.elements,
            view_proj,
            // w: 1 with reversed depth — the sky pin and the screen-space
            // marches read it.
            camera_position: [
                camera.position().x,
                camera.position().y,
                camera.position().z,
                if reversed { 1.0 } else { 0.0 },
            ],
            fog_color: [scene.fog.color.r, scene.fog.color.g, scene.fog.color.b, 1.0],
            fog_params: [
                scene.fog.near,
                scene.fog.far,
                scene.fog.density,
                scene.fog.mode as f32,
            ],
            tone_mapping_exposure: [
                if self.tone_map_pass && !self.cube_pass {
                    self.tone_mapping.shader_mode()
                } else {
                    0.0
                },
                self.exposure,
                if has_env { 1.0 } else { 0.0 },
                // 2: linear because the target is sRGB and encodes on write,
                // which the shader's data outputs (`framebuffer_data`) undo.
                if !shader_outputs_linear {
                    0.0
                } else if !linear_framebuffer && self.color_format.is_srgb() {
                    2.0
                } else {
                    1.0
                },
            ],
            viewport_size: [
                self.depth_size.0 as f32,
                self.depth_size.1 as f32,
                cam_near,
                cam_far,
            ],
            ..FrameUniforms::default()
        };

        let mut ambient = [0.0_f32; 3];
        let mut n_dir = 0usize;
        let mut n_point = 0usize;
        let mut n_spot = 0usize;
        let mut n_hemi = 0usize;
        let mut n_rect = 0usize;
        let oit_available = self.pipeline_oit.is_some();
        // Every shadow-casting light, by kind: its index in the kind's light
        // array, and what its shadow pass needs.
        // Every shadow-casting light, in the order its record takes. A light
        // names its record in its own `color[3]`.
        let mut casters: Vec<Caster> = Vec::new();

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Topology {
            Triangle,
            TriangleAlpha,
            TriangleWire,
            TriangleNoCull,
            Sky,
            Line,
            /// three.js `Points`: one instance per point, each a quad
            /// (`vs_points`).
            Point,
            /// `Point` with a transparent material: blended and sorted with
            /// the other transparent draws.
            PointAlpha,
            /// `LineSegments` wider than a pixel: one instance per segment,
            /// each a round-capped quad (`vs_wide_lines`).
            WideLine,
            /// `WideLine` blended: sorted with the transparent draws, its
            /// edges faded by coverage.
            WideLineAlpha,
            Sprite,
            Instanced,
            /// `Instanced` with a `wireframe: true` material: line list.
            InstancedWire,
            /// `Instanced` with a transparent material: alpha-blended, sorted
            /// with the other transparent draws, as three.js renders a
            /// transparent `InstancedMesh`.
            InstancedAlpha,
            Skinned,
            /// `Skinned` with a `wireframe: true` material: line list.
            SkinnedWire,
            /// `Skinned` with a transparent material.
            SkinnedAlpha,
            /// Single-layer glass depth prepass (writes depth, no color).
            GlassDepth,
            /// The tint half of the glass colour pass, drawn just before
            /// `GlassColor`: what is behind times the glass's transmittance.
            GlassTint,
            /// Single-layer glass color pass (LessEqual, no depth write): the
            /// light half, added over the tint.
            GlassColor,
            /// Weighted-blended OIT surface (drawn into accum+revealage MRT).
            Oit,
            /// Screen-space refraction glass (drawn in a dedicated pass that
            /// samples the captured opaque color + depth).
            Refract,
            /// Custom-shader material:
            /// (fragment-source hash, is-transparent, wants-screen-space).
            /// The last flag moves the draw into the screen-space pass, where the
            /// captured opaque colour and depth are bound at `@group(3)`.
            /// (shader hash, transparent, screen-space, side) — exactly the
            /// custom pipeline cache key minus the framebuffer format, so
            /// every lookup site can build the key from the topology alone.
            Custom(u64, bool, bool, u32),
        }
        impl Topology {
            /// A plain triangle mesh that a glass mode draws in passes of its
            /// own: in the shadow maps it is the triangles it always was.
            fn is_glass_mesh(self) -> bool {
                matches!(self, Topology::Refract | Topology::Oit | Topology::GlassColor)
            }
            /// Drawn from the mesh's wire (line-list) index buffer.
            fn is_wire(self) -> bool {
                matches!(
                    self,
                    Topology::TriangleWire | Topology::InstancedWire | Topology::SkinnedWire
                )
            }
            /// Reads per-instance matrices from vertex slot 1.
            fn is_instanced(self) -> bool {
                matches!(
                    self,
                    Topology::Instanced | Topology::InstancedWire | Topology::InstancedAlpha
                )
            }
            /// Binds the combined mesh+skin group and joint attributes.
            fn is_skinned(self) -> bool {
                matches!(self, Topology::Skinned | Topology::SkinnedWire | Topology::SkinnedAlpha)
            }
            /// Drawn as faces, so three.js's `side` culls them (`side_cull`).
            fn is_faces(self) -> bool {
                matches!(
                    self,
                    Topology::Triangle
                        | Topology::TriangleNoCull
                        | Topology::TriangleAlpha
                        | Topology::Instanced
                        | Topology::InstancedAlpha
                        | Topology::Skinned
                        | Topology::SkinnedAlpha
                )
            }
            /// One instance per point, a quad each, read from the geometry's
            /// own vertex buffer.
            fn is_points(self) -> bool {
                matches!(self, Topology::Point | Topology::PointAlpha)
            }
            /// One instance per line segment, read two vertices at a time.
            fn is_wide_lines(self) -> bool {
                matches!(self, Topology::WideLine | Topology::WideLineAlpha)
            }
            /// Blends with what is behind it — and so, on an sRGB target,
            /// draws in the encoded-blend pass (see `encoded_pass`). Glass
            /// included, prepass and all: it sorts with the transparent draws
            /// by `renderOrder`, so it has to be drawn in the same pass as
            /// them or a glass shell ordered after them still went first.
            fn encodes(self) -> bool {
                matches!(
                    self,
                    Topology::GlassDepth
                        | Topology::GlassTint
                        | Topology::GlassColor
                        | Topology::TriangleAlpha
                        | Topology::PointAlpha
                        | Topology::WideLineAlpha
                        | Topology::Sprite
                        | Topology::InstancedAlpha
                        | Topology::SkinnedAlpha
                        | Topology::Custom(_, true, false, ..)
                )
            }
        }
        #[derive(Clone)]
        struct DrawMesh {
            key: *const BufferGeometry,
            topology: Topology,
            /// Z in view space, used to back-to-front sort transparent draws.
            view_z: f32,
            render_order: i32,
            /// False when the camera frustum cannot see this. It still has to
            /// be drawn into the shadow maps — a caster behind the camera can
            /// throw a shadow across the whole view — so culled draws stay in
            /// this list and only the colour passes skip them.
            in_frustum: bool,
            /// three.js's transmissive material (`Material::transmissive`).
            transmissive: bool,
            /// Any transmissive physical material: cast with the cut-out shadow
            /// pipelines, which drop texels for the light it lets through.
            shadow_transmits: bool,
            cast_shadow: bool,
            refract_capture: bool,
            alpha_test: f32,
            /// The material's three.js `side` (0 front, 1 back, 2 double): the
            /// shadow passes cull by it, as three.js's `shadowSide` does.
            side: u32,
            /// The `side` the shadow passes use: the material's, even when the
            /// colour passes draw a blended DoubleSide mesh as its back and
            /// front halves (see the two-pass split before the sort).
            shadow_side: u32,
            model: [f32; 16],
            normal_matrix: [f32; 16],
            color: [f32; 4],
            emissive: [f32; 4],
            specular: [f32; 4],
            shininess_or_near: f32,
            opacity_or_far: f32,
            roughness: f32,
            metalness: f32,
            ao_intensity: f32,
            normal_scale: [f32; 2],
            physical: [f32; 4],
            physical2: [f32; 4],
            physical3: [f32; 4],
            physical4: [f32; 4],
            physical5: [f32; 4],
            displacement: [f32; 4],
            uv_transform: [f32; 4],
            atmos: [f32; 4],
            twilight: [f32; 4],
            eclipse: [f32; 4],
            scatter: [f32; 4],
            aurora: [f32; 4],
            shader_flags: u32,
            kind: MaterialKind,
            slots: MaterialTextureSlots,
            /// The material shader variant this draw needs, if its maps do not
            /// fit the legacy layout (see `material_maps`).
            map_variant: Option<super::material_maps::MapVariant>,
            /// `Material::map_uniforms`, a variant's uniform block.
            map_uniforms: [[f32; 4]; 6],
            /// A Standard/Physical material's own environment
            /// (`material_env_groups`), in place of the scene's.
            env_group: Option<MaterialEnvKey>,
            /// Index into `instance_buffers` for Topology::Instanced.
            instance_buf_idx: usize,
            /// The instance data's content stamp (`InstanceBuffer::stamp`),
            /// for the shadow fingerprint; 0 for a plain mesh.
            instance_stamp: u64,
            instance_count: u32,
            /// Index into `skin_data` for Topology::Skinned.
            skin_idx: usize,
            /// Index into `shader_data` for Topology::Custom.
            shader_idx: usize,
            /// The part of the index (vertices, unindexed) this draw covers —
            /// a group and/or the draw range, three.js's `renderBufferDirect`
            /// clipping. `None` is the whole geometry.
            range: Option<(u32, u32)>,
            /// Morph uniforms (count 0: no morphing) and the texture to bind.
            morph: [f32; 4],
            morph_weights: [f32; crate::core::MAX_MORPH_TARGETS],
            morph_tex: (usize, u32),
            /// three.js `polygonOffset`: (factor bits, units), drawn with a
            /// depth-biased pipeline (see `Renderer::biased_pipelines`).
            polygon_offset: Option<(u32, i32)>,
            /// three.js `depthTest` / `depthWrite` / `depthFunc` /
            /// `colorWrite`, applied over whichever pipeline the draw picks
            /// (see `pipeline_state`). Always the default for the glass,
            /// order-independent and refraction modes, which own theirs.
            state: crate::materials::RenderState,
            /// `Object3D::clip_plane`, packed for `MeshUniforms`.
            clip_plane: [f32; 4],
        }
        let mut draws: Vec<DrawMesh> = Vec::new();
        // The triangle topologies a three.js `polygonOffset` applies to, as
        // the code `build_biased_pipeline` builds them from: plain (back-face
        // culled), alpha-blended, double-sided.
        let biased_topology = |t: Topology| -> Option<u8> {
            match t {
                Topology::Triangle => Some(0),
                Topology::TriangleAlpha => Some(1),
                Topology::TriangleNoCull => Some(2),
                _ => None,
            }
        };
        // Per-custom-shader-material draw data (parallel to Topology::Custom).
        struct ShaderDrawData {
            fragment: String,
            vertex: Option<String>,
            data: [[f32; 4]; 16],
            storage: [Vec<f32>; 3],
            textures: Vec<std::sync::Arc<wgpu::TextureView>>,
            volumes: Vec<std::sync::Arc<wgpu::TextureView>>,
            /// Drawn from an `InstancedMesh`: the instanced pipeline layout,
            /// with the instance buffer at `instance_buf_idx` in slot 1 and
            /// the custom columns at `custom_buf` in slot 2.
            instanced: bool,
            custom_buf: usize,
            /// three.js `alphaToCoverage`.
            alpha_to_coverage: bool,
        }
        let mut shader_data: Vec<ShaderDrawData> = Vec::new();
        let mut instance_buffers: Vec<wgpu::Buffer> = Vec::new();
        struct SkinData {
            bone_buf: wgpu::Buffer,
            bg: Option<wgpu::BindGroup>,
            /// The bone matrices, hashed, for the shadow fingerprint.
            bone_hash: u64,
        }
        let mut skin_data: Vec<SkinData> = Vec::new();

        // Helper: build a DrawMesh from a geometry+material pair under a given topology.
        let make_draw = |geom: &std::sync::Arc<BufferGeometry>,
                         mat: &Material,
                         obj: &crate::core::Object3D,
                         topology: Topology|
         -> DrawMesh {
            let key = geom_cache_key(geom);
            let nm3 = Matrix3::normal_matrix(&obj.matrix_world);
            // Mirror materials repurpose the normal_matrix slot for the
            // projective texture matrix supplied per-frame from JS.
            let normal_matrix = match mat {
                Material::Mirror(m) => m.texture_matrix,
                _ => mat3_to_mat4_array(&nm3),
            };
            // Promote Triangle to TriangleAlpha / TriangleWire based on material
            // flags. Pre-set view_z for later sort.
            let world_pos = obj.world_position();
            let view_z = view_m.elements[2] * world_pos.x
                + view_m.elements[6] * world_pos.y
                + view_m.elements[10] * world_pos.z
                + view_m.elements[14];
            let topology = if topology == Topology::Instanced && mat.wireframe() {
                Topology::InstancedWire
            } else if topology == Topology::Instanced && mat.transparent() {
                Topology::InstancedAlpha
            } else if topology == Topology::Skinned && mat.wireframe() {
                Topology::SkinnedWire
            } else if topology == Topology::Skinned && mat.transparent() {
                Topology::SkinnedAlpha
            } else if topology == Topology::Triangle {
                if matches!(mat, Material::Sky(_)) {
                    Topology::Sky
                } else if mat.wireframe() {
                    Topology::TriangleWire
                } else if mat.transparent() {
                    Topology::TriangleAlpha
                } else if mat.side() != 0 {
                    Topology::TriangleNoCull
                } else {
                    Topology::Triangle
                }
            } else {
                topology
            };
            let c = mat.color();
            let e = mat.emissive();
            // For MeshDistanceMaterial, pack the reference position into the
            // specular channel (unused by that path) so the shader can read it.
            // For SkyMaterial, pack the sun position into specular.
            let s = match mat {
                Material::Distance(m) => crate::math::Color::new(
                    m.reference_position.x,
                    m.reference_position.y,
                    m.reference_position.z,
                ),
                Material::Sky(m) => {
                    crate::math::Color::new(m.sun_position.x, m.sun_position.y, m.sun_position.z)
                }
                _ => mat.specular(),
            };
            // For SkyMaterial, pack atmospheric tuning into params.x/y (replacing
            // shininess/opacity which it doesn't use).
            let (shin_or_near, opacity_or_far) = match mat {
                Material::Depth(m) => (m.near, m.far),
                Material::Distance(m) => (m.near_distance, m.far_distance),
                Material::Sky(m) => (m.turbidity, m.rayleigh),
                _ => (mat.shininess(), mat.opacity()),
            };
            let physical = match mat {
                Material::Physical(m) => {
                    [m.clearcoat, m.clearcoat_roughness, m.ior, m.transmission]
                }
                Material::Line(m) => [m.dash_scale, m.dash_size, m.gap_size, 0.0],
                // The atmosphere shader intersects the view ray with both
                // spheres, so it needs their radii rather than a surface
                // description.
                Material::Atmosphere(m) => {
                    [m.planet_radius, m.atmosphere_radius, m.intensity, m.falloff]
                }
                _ => [0.0; 4],
            };
            let physical2 = match mat {
                Material::Physical(m) => [
                    m.thickness,
                    m.dispersion,
                    m.vertex_emissive,
                    m.anisotropy_rotation,
                ],
                Material::Atmosphere(m) => [
                    m.sunset_color.r,
                    m.sunset_color.g,
                    m.sunset_color.b,
                    m.airglow,
                ],
                _ => [0.0; 4],
            };
            let physical3 = match mat {
                Material::Physical(m) => [
                    m.iridescence,
                    m.iridescence_ior,
                    m.iridescence_thickness_range[1],
                    m.sheen,
                ],
                Material::Atmosphere(m) => {
                    [m.airglow_color.r, m.airglow_color.g, m.airglow_color.b, 0.0]
                }
                _ => [0.0; 4],
            };
            let physical4 = match mat {
                Material::Physical(m) => [
                    m.sheen_color.r,
                    m.sheen_color.g,
                    m.sheen_color.b,
                    m.sheen_roughness,
                ],
                _ => [0.0; 4],
            };
            let physical5 = match mat {
                Material::Physical(m) => {
                    // attenuation_distance defaults to INFINITY. Feeding that to
                    // the GPU makes the Beer-Lambert coefficient -log(c)/inf,
                    // which is 0 for c>0 but NaN for a fully-black channel — and
                    // one NaN poisons the whole fragment. Collapse any
                    // non-finite/non-positive distance to 0, the shader's
                    // "absorption disabled" sentinel.
                    let dist = if m.attenuation_distance.is_finite() && m.attenuation_distance > 0.0
                    {
                        m.attenuation_distance
                    } else {
                        0.0
                    };
                    [
                        m.attenuation_color.r,
                        m.attenuation_color.g,
                        m.attenuation_color.b,
                        dist,
                    ]
                }
                _ => [0.0; 4],
            };
            let displacement = {
                let (scale, bias) = mat.displacement();
                let night = mat.emissive_night_side();
                [scale, bias, night, 0.0]
            };
            // three.js gives every map its own matrix. The legacy layout
            // holds one scale and translation, applied in the vertex stage to
            // every map; a material whose maps need more gets a shader variant
            // that transforms each map itself (`material_maps::needs_variant`).
            let uv_transform = super::material_maps::legacy_uv_transform(&mat.texture_slots());
            let atmos = mat.atmosphere_params();
            let (eclipse, sun_angle) = mat.eclipse();
            let twilight = {
                let c = mat.twilight_color();
                [c.r, c.g, c.b, sun_angle]
            };
            let scatter = mat.scatter_params();
            let aurora = mat.aurora_params();
            let mut shader_flags = match mat {
                Material::Line(m) if m.dash_size > 0.0 || m.gap_size > 0.0 => FLAG_DASHED,
                Material::Basic(m) if m.shadow_only => FLAG_SHADOW_MAT,
                _ => 0,
            };
            if obj.receive_shadow {
                shader_flags |= FLAG_RECEIVE_SHADOW;
            }
            if !mat.tone_mapped() {
                shader_flags |= FLAG_NO_TONE_MAP;
            }
            let shadow_transmits = matches!(mat, Material::Physical(m)
                if m.transmission > 0.0
                    && (m.thickness <= 0.0 || m.transparency == crate::materials::TransparencyMode::Glass));
            if shadow_transmits {
                shader_flags |= FLAG_SHADOW_TRANSMITS;
            }
            if let Material::Physical(m) = mat {
                if m.transmission > 0.0
                    && matches!(
                        m.transparency,
                        crate::materials::TransparencyMode::Glass | crate::materials::TransparencyMode::Oit
                    )
                {
                    shader_flags |= FLAG_BLENDED_GLASS;
                }
            }
            DrawMesh {
                key,
                topology,
                view_z,
                render_order: obj.render_order,
                clip_plane: clip_plane_of(obj),
                in_frustum: true,
                transmissive: mat.transmissive(),
                shadow_transmits,
                cast_shadow: obj.cast_shadow,
                refract_capture: obj.refract_capture,
                alpha_test: mat.alpha_test(),
                side: mat.side(),
                shadow_side: mat.side(),
                model: obj.matrix_world.elements,
                normal_matrix,
                color: [c.r, c.g, c.b, mat.opacity()],
                // emissive.w carries the physical-material anisotropy strength
                // (0 = isotropic); the shader stretches the specular lobe along
                // the surface tangent by this amount. Non-physical mats leave 0.
                emissive: [
                    e.r,
                    e.g,
                    e.b,
                    match mat {
                        Material::Physical(m) => m.anisotropy,
                        _ => 0.0,
                    },
                ],
                // Phong's specular colour; Physical's `specularColor` and, in
                // w, `specularIntensity`.
                specular: match mat {
                    Material::Physical(m) => {
                        let c = m.specular_color;
                        [c.r, c.g, c.b, m.specular_intensity]
                    }
                    _ => [s.r, s.g, s.b, 0.0],
                },
                shininess_or_near: shin_or_near,
                opacity_or_far,
                // For SkyMaterial, repurpose roughness/metalness as
                // params.z/w (mieCoefficient/mieDirectionalG).
                roughness: match mat {
                    Material::Sky(m) => m.mie_coefficient,
                    _ => mat.roughness(),
                },
                metalness: match mat {
                    Material::Sky(m) => m.mie_directional_g,
                    _ => mat.metalness(),
                },
                ao_intensity: mat.ao_intensity(),
                normal_scale: [mat.normal_scale().x, mat.normal_scale().y],
                physical,
                physical2,
                physical3,
                physical4,
                physical5,
                displacement,
                uv_transform,
                atmos,
                twilight,
                eclipse,
                scatter,
                aurora,
                shader_flags,
                kind: mat.kind(),
                slots: mat.texture_slots(),
                map_variant: None,
                map_uniforms: mat.map_uniforms(),
                env_group: mat.pbr_env().map(material_env_key),
                instance_buf_idx: usize::MAX,
                instance_stamp: 0,
                instance_count: 1,
                skin_idx: usize::MAX,
                shader_idx: usize::MAX,
                range: None,
                morph: [0.0; 4],
                morph_weights: [0.0; crate::core::MAX_MORPH_TARGETS],
                morph_tex: (0, 0),
                polygon_offset: mat.polygon_offset().map(|o| (o.factor.to_bits(), o.units.round() as i32)),
                state: mat.render_state(),
            }
        };

        let cam_layers = camera.layers();
        let root = scene.root;
        self.instance_frame += 1;
        // Meshes not drawn for a while (removed, hidden, culled away) give
        // their buffers back. Generous, as a frame of several panes renders
        // more than once.
        let now = self.instance_frame;
        self.instance_cache.retain(|_, c| now - c.frame < 64);
        scene.arena.traverse_visible(root, &mut |obj_id, obj| {
            if !cam_layers.test(&obj.layers) {
                return;
            }
            match &obj.kind {
                ObjectKind::Light(light) => match light {
                    Light::Ambient(l) => {
                        ambient[0] += l.color.r * l.intensity;
                        ambient[1] += l.color.g * l.intensity;
                        ambient[2] += l.color.b * l.intensity;
                    }
                    Light::Directional(l) => {
                        if n_dir >= MAX_DIR_LIGHTS {
                            return;
                        }
                        // three.js: direction = normalize(target - position); default target (0,0,0).
                        let pos = obj.world_position();
                        let dir = {
                            let d = Vector3::new(-pos.x, -pos.y, -pos.z);
                            if d.length_sq() > 1e-8 {
                                d.normalize()
                            } else {
                                transform_direction(&obj.matrix_world, l.direction).normalize()
                            }
                        };
                        let c = l.color;
                        frame_u.dir_lights[n_dir] = DirLightGpu {
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Dir { dir, settings: l.shadow });
                            frame_u.dir_lights[n_dir].color[3] = casters.len() as f32;
                        }
                        n_dir += 1;
                    }
                    Light::Point(l) => {
                        if n_point >= MAX_POINT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let c = l.color;
                        frame_u.point_lights[n_point] = PointLightGpu {
                            position: [p.x, p.y, p.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            params: [l.distance, l.decay, 0.0, 0.0],
                        };
                        if l.cast_shadow {
                            // three.js `PointLightShadow`: `light.distance ||
                            // camera.far`.
                            let radius = if l.distance > 0.0 { l.distance } else { l.shadow.camera_far };
                            if casters.len() < MAX_SHADOW_RECORDS {
                                casters.push(Caster::Point { pos: p, radius, settings: l.shadow });
                                frame_u.point_lights[n_point].color[3] = casters.len() as f32;
                            }
                        }
                        n_point += 1;
                    }
                    Light::Spot(l) => {
                        if n_spot >= MAX_SPOT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let dir = transform_direction(&obj.matrix_world, l.direction).normalize();
                        let c = l.color;
                        let cos_outer = l.angle.cos();
                        let cos_inner = (l.angle * (1.0 - l.penumbra)).cos();
                        frame_u.spot_lights[n_spot] = SpotLightGpu {
                            position: [p.x, p.y, p.z, 0.0],
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            params: [l.distance, l.decay, cos_outer, cos_inner],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Spot { pos: p, dir, angle: l.angle, settings: l.shadow });
                            frame_u.spot_lights[n_spot].color[3] = casters.len() as f32;
                        }
                        n_spot += 1;
                    }
                    Light::Hemisphere(l) => {
                        if n_hemi >= MAX_HEMI_LIGHTS {
                            return;
                        }
                        let up = transform_direction(&obj.matrix_world, Vector3::UP).normalize();
                        frame_u.hemi_lights[n_hemi] = HemiLightGpu {
                            sky_color: [
                                l.sky_color.r * l.intensity,
                                l.sky_color.g * l.intensity,
                                l.sky_color.b * l.intensity,
                                0.0,
                            ],
                            ground_color: [
                                l.ground_color.r * l.intensity,
                                l.ground_color.g * l.intensity,
                                l.ground_color.b * l.intensity,
                                0.0,
                            ],
                            direction: [up.x, up.y, up.z, 0.0],
                        };
                        n_hemi += 1;
                    }
                    Light::RectArea(l) => {
                        if n_rect >= MAX_RECT_LIGHTS {
                            return;
                        }
                        // three.js orients a RectAreaLight by its matrixWorld:
                        // width runs along local +X, height along local +Y, and
                        // the emitting face looks down local -Z. Store half
                        // extents so the shader can form corners by ± directly.
                        // Through the world matrix's rotation alone, as
                        // three.js's `extractRotation`: a scaled light keeps
                        // its `width` and `height`.
                        let pos = obj.world_position();
                        let unit = |v: Vector3| {
                            let d = transform_direction(&obj.matrix_world, v);
                            let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-12);
                            Vector3::new(d.x / len, d.y / len, d.z / len)
                        };
                        let (ax, ay) = (unit(Vector3::new(1.0, 0.0, 0.0)), unit(Vector3::new(0.0, 1.0, 0.0)));
                        let right = Vector3::new(ax.x * l.width * 0.5, ax.y * l.width * 0.5, ax.z * l.width * 0.5);
                        let up = Vector3::new(ay.x * l.height * 0.5, ay.y * l.height * 0.5, ay.z * l.height * 0.5);
                        let c = l.color;
                        frame_u.rect_lights[n_rect] = RectLightGpu {
                            position: [pos.x, pos.y, pos.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            right: [right.x, right.y, right.z, 0.0],
                            up: [up.x, up.y, up.z, 0.0],
                        };
                        n_rect += 1;
                    }
                },
                ObjectKind::Mesh(mesh) => {
                    self.ensure_geometry(&mesh.geometry);
                    // Morph targets: the per-draw uniforms and the geometry's
                    // morph texture, shared by every group of the mesh.
                    let morph = crate::core::morph_uniforms(&mesh.geometry, &mesh.morph_influences)
                        .and_then(|u| Some((u, self.ensure_morph(&mesh.geometry)?)));
                    let full = mesh.geometry.draw_count() as u32;
                    // One draw per three.js render item: the whole draw range
                    // with one material, or each group with its own — opaque
                    // and transparent sorted per group, as three.js's render
                    // lists are.
                    for item in mesh.draws() {
                    let material = item.material;
                    let range = (item.start != 0 || item.end != full).then_some((item.start, item.end));
                    if material.wireframe() {
                        match range {
                            None => self.ensure_wire_indices(&mesh.geometry),
                            Some((a, b)) => self.ensure_wire_span(&mesh.geometry, a, b),
                        }
                    }
                    let with_span = |mut d: DrawMesh| -> DrawMesh {
                        d.range = range;
                        if let Some(((header, weights), tex)) = morph {
                            d.morph = header;
                            d.morph_weights = weights;
                            d.morph_tex = tex;
                        }
                        d
                    };
                    let mut d = with_span(make_draw(&mesh.geometry, material, obj, Topology::Triangle));
                    // Custom-shader material: route to a per-source pipeline and
                    // stash its user data for the @group(4) bind group.
                    if let Material::Shader(sm) = &**material {
                        // Morph targets reach a shader material as three.js's
                        // `#include <morphtarget_vertex>` does: the default
                        // vertex stage (`standard_vertex_morphed`) applies
                        // them, and a custom one calls `morph_position` /
                        // `morph_normal` / `morph_color` itself. Its group 2
                        // then carries the geometry's morph texture.
                        let hash = custom_shader_hash(&sm.fragment, sm.vertex.as_deref())
                            ^ if sm.alpha_to_coverage { COVERAGE_CUSTOM_SALT } else { 0 };
                        let transparent = sm.transparent || sm.opacity < 1.0;
                        d.topology = Topology::Custom(
                            hash,
                            transparent,
                            sm.screen_space,
                            sm.side,
                        );
                        d.shader_idx = shader_data.len();
                        let pad = |s: &Vec<f32>| {
                            if s.is_empty() {
                                vec![0.0f32]
                            } else {
                                s.clone()
                            }
                        };
                        shader_data.push(ShaderDrawData {
                            fragment: sm.fragment.clone(),
                            vertex: sm.vertex.clone(),
                            data: sm.data_slots(),
                            storage: [pad(&sm.storage0), pad(&sm.storage1), pad(&sm.storage2)],
                            textures: sm.textures.clone(),
                            volumes: sm.volumes.clone(),
                            instanced: false,
                            custom_buf: 0,
                            alpha_to_coverage: sm.alpha_to_coverage,
                        });
                    }
                    // Route transparent PhysicalMaterials to single-layer glass
                    // (depth prepass + color) or weighted-blended OIT.
                    if d.topology == Topology::TriangleAlpha {
                        // Where OIT cannot run (WebGL2 blends every attachment
                        // alike), its glass draws as `Glass` does — tinted,
                        // lit and shadowed, one layer deep — rather than not
                        // at all.
                        let mode = match material.transparency_mode() {
                            crate::materials::TransparencyMode::Oit if !oit_available => {
                                crate::materials::TransparencyMode::Glass
                            }
                            m => m,
                        };
                        match mode {
                            crate::materials::TransparencyMode::Glass => {
                                let mut prepass = with_span(make_draw(
                                    &mesh.geometry,
                                    material,
                                    obj,
                                    Topology::GlassDepth,
                                ));
                                prepass.state = crate::materials::RenderState::DEFAULT;
                                draws.push(prepass);
                                // The tint half, then (as `d`) the light half.
                                let mut tint = d.clone();
                                tint.topology = Topology::GlassTint;
                                tint.state = crate::materials::RenderState::DEFAULT;
                                tint.cast_shadow = false;
                                draws.push(tint);
                                d.topology = Topology::GlassColor;
                            }
                            crate::materials::TransparencyMode::Oit => d.topology = Topology::Oit,
                            crate::materials::TransparencyMode::Refract => {
                                d.topology = Topology::Refract
                            }
                            crate::materials::TransparencyMode::Blend => {}
                        }
                        // These modes own their depth state (a prepass, an
                        // accumulation target); the material's does not apply.
                        if d.topology != Topology::TriangleAlpha {
                            d.state = crate::materials::RenderState::DEFAULT;
                        }
                    }
                    draws.push(d);
                    }
                }
                ObjectKind::LineSegments(ls) => {
                    self.ensure_geometry(&ls.geometry);
                    // three.js applies the draw range to lines too.
                    let full = ls.geometry.draw_count() as u32;
                    let Some((start, end)) = ls.geometry.draw_span(None) else {
                        return;
                    };
                    // three.js `linewidth`, which WebGL ignores: wider than
                    // a pixel, a capsule a segment (`vs_wide_lines`), its
                    // width in device pixels as a point's size is
                    let width = match &*ls.material {
                        Material::Line(m) => m.line_width,
                        _ => 1.0,
                    };
                    // Thinner than a pixel and blended, a capsule too: each
                    // pixel takes the share of it the line covers, where a
                    // line primitive lights whole pixels or none. Opaque, a
                    // capsule could only cut at half, so it stays a line.
                    let blended = ls.material.transparent();
                    if width > 1.0 || (blended && width > 0.0 && width < 1.0) {
                        let key = geom_cache_key(&ls.geometry);
                        if let Some(e) = self.geom_cache.get_mut(&key) {
                            e.mesh.ensure_segments(&self.device, &ls.geometry);
                        }
                        let topology = if blended { Topology::WideLineAlpha } else { Topology::WideLine };
                        let mut d = make_draw(&ls.geometry, &ls.material, obj, topology);
                        d.range = (start != 0 || end != full).then_some((start, end));
                        d.shininess_or_near = width * self.pixel_ratio();
                        d.roughness = if blended { 1.0 } else { 0.0 };
                        d.cast_shadow = false;
                        draws.push(d);
                        return;
                    }
                    let mut d = make_draw(&ls.geometry, &ls.material, obj, Topology::Line);
                    d.range = (start != 0 || end != full).then_some((start, end));
                    draws.push(d);
                }
                ObjectKind::Points(p) => {
                    // three.js `Points`: one draw, one instance per point,
                    // each expanded to a `size`-pixel quad by `vs_points`
                    // (WebGPU has no point size). `params.x` carries the size
                    // in device pixels (× the pixel ratio, as three.js's
                    // `size` uniform), `params.w` whether it attenuates.
                    self.ensure_geometry(&p.geometry);
                    let full = p.geometry.draw_count() as u32;
                    let Some((start, end)) = p.geometry.draw_span(None) else {
                        return;
                    };
                    let topology = if p.material.transparent() {
                        Topology::PointAlpha
                    } else {
                        Topology::Point
                    };
                    let mut d = make_draw(&p.geometry, &p.material, obj, topology);
                    d.range = (start != 0 || end != full).then_some((start, end));
                    d.shininess_or_near = p.material.point_size() * self.pixel_ratio();
                    d.metalness = p.material.point_size_attenuation() as u32 as f32;
                    d.cast_shadow = false;
                    draws.push(d);
                }
                ObjectKind::Sprite(sprite) => {
                    // Inline construction — the sprite quad is built-in, so we
                    // can't reuse make_draw (which keys off an Arc<BufferGeometry>).
                    let mat = &*sprite.material;
                    // three.js `sprite_vert` uniforms, in slots the sprite
                    // shader has to itself (see `vs_sprite`): the centre as
                    // an offset from the middle, so that zero — what point
                    // draws leave there — is three.js's default (0.5, 0.5),
                    // and the rotation; `params13.x` set means size
                    // attenuation off.
                    let (rotation, size_attenuation) = match mat {
                        Material::Sprite(m) => (m.rotation, m.size_attenuation),
                        _ => (0.0, true),
                    };
                    let sprite_params =
                        [sprite.center.x - 0.5, sprite.center.y - 0.5, 0.0, rotation];
                    let nm3 = Matrix3::normal_matrix(&obj.matrix_world);
                    let c = mat.color();
                    let sp = obj.world_position();
                    let svz = view_m.elements[2] * sp.x
                        + view_m.elements[6] * sp.y
                        + view_m.elements[10] * sp.z
                        + view_m.elements[14];
                    draws.push(DrawMesh {
                        key: std::ptr::null::<BufferGeometry>(),
                        topology: Topology::Sprite,
                        view_z: svz,
                        render_order: obj.render_order,
                        clip_plane: clip_plane_of(obj),
                        in_frustum: true,
                        transmissive: false,
                        shadow_transmits: false,
                        cast_shadow: false,
                        refract_capture: true,
                        alpha_test: mat.alpha_test(),
                        side: mat.side(),
                        shadow_side: mat.side(),
                        model: obj.matrix_world.elements,
                        normal_matrix: mat3_to_mat4_array(&nm3),
                        color: [c.r, c.g, c.b, mat.opacity()],
                        emissive: [0.0; 4],
                        specular: [0.0; 4],
                        shininess_or_near: 0.0,
                        opacity_or_far: mat.opacity(),
                        roughness: 1.0,
                        metalness: 0.0,
                        ao_intensity: 1.0,
                        normal_scale: [1.0, 1.0],
                        physical: [0.0; 4],
                        physical2: [0.0; 4],
                        physical3: [0.0; 4],
                        physical4: [0.0; 4],
                        physical5: [0.0; 4],
                        displacement: sprite_params,
                        uv_transform: [0.0, 0.0, 1.0, 1.0],
                        atmos: [0.0; 4],
                        twilight: [0.0; 4],
                        eclipse: [0.0; 4],
                        scatter: [if size_attenuation { 0.0 } else { 1.0 }, 0.0, 0.0, 0.0],
                        aurora: [0.0; 4],
                        shader_flags: (if mat.tone_mapped() { 0 } else { FLAG_NO_TONE_MAP })
                            | if mat.transparent() { 0 } else { FLAG_OPAQUE_OUT },
                        kind: mat.kind(),
                        slots: mat.texture_slots(),
                        map_variant: None,
                        map_uniforms: mat.map_uniforms(),
                env_group: mat.pbr_env().map(material_env_key),
                        instance_buf_idx: usize::MAX,
                instance_stamp: 0,
                        instance_count: 1,
                        skin_idx: usize::MAX,
                        shader_idx: usize::MAX,
                        range: None,
                        morph: [0.0; 4],
                        morph_weights: [0.0; crate::core::MAX_MORPH_TARGETS],
                        morph_tex: (0, 0),
                        polygon_offset: None,
                        state: mat.render_state(),
                    });
                }
                ObjectKind::InstancedMesh(im) => {
                    self.ensure_geometry(&im.geometry);
                    if im.transforms.is_empty() {
                        return;
                    }
                    // Its draws: the draw range with one material, or every
                    // group with its own from the material array — each one
                    // instanced, as three.js's render list holds an
                    // InstancedMesh with a material array.
                    let items: Vec<DrawItem<std::sync::Arc<Material>>> = {
                        let full = im.geometry.draw_count() as u32;
                        im.draws()
                            .into_iter()
                            .map(|i| {
                                (
                                    i.material.clone(),
                                    (i.start != 0 || i.end != full).then_some((i.start, i.end)),
                                )
                            })
                            .collect()
                    };
                    if items.is_empty() {
                        return;
                    }
                    for (material, range) in &items {
                        if material.wireframe() {
                            match range {
                                None => self.ensure_wire_indices(&im.geometry),
                                Some((a, b)) => self.ensure_wire_span(&im.geometry, *a, *b),
                            }
                        }
                    }
                    // Morph targets: one row per instance when the mesh has a
                    // morph texture (`set_morph_at`), else the shared
                    // influences as for a mesh.
                    let row_len = im.morph_row_len();
                    let per_instance = !im.morph_texture.is_empty()
                        && im.morph_texture.len() == row_len * im.transforms.len();
                    let morph = if per_instance {
                        crate::core::morph_header(&im.geometry).and_then(|mut h| {
                            h[3] = (h[3] as u32 | 16) as f32;
                            let tex = self.ensure_instance_morph(
                                im as *const crate::core::InstancedMesh as usize,
                                &im.geometry,
                                &im.morph_texture,
                                im.transforms.len() as u32,
                                row_len as u32,
                            )?;
                            Some(((h, [0.0; crate::core::MAX_MORPH_TARGETS]), tex))
                        })
                    } else {
                        crate::core::morph_uniforms(&im.geometry, &im.morph_influences)
                            .and_then(|u| Some((u, self.ensure_morph(&im.geometry)?)))
                    };
                    // Twenty floats an instance: the matrix, then the tint.
                    // Instances past the end of `colors` are white, so a caller
                    // that never sets one pays four floats and nothing else.
                    // A mesh with an `instance_version` that has not moved
                    // keeps last frame's buffer without being read at all.
                    let version = im.instance_version.map(|v| (v, im.transforms.len()));
                    let buf = match self.cached_instances(obj_id, 0, version) {
                        Some(b) => b,
                        None => {
                            let mut matrices: Vec<f32> = Vec::with_capacity(im.transforms.len() * 20);
                            for (i, t) in im.transforms.iter().enumerate() {
                                matrices.extend_from_slice(&t.elements);
                                let c = im.colors.get(i).copied().unwrap_or(crate::math::Color::WHITE);
                                matrices.extend_from_slice(&[c.r, c.g, c.b, 1.0]);
                            }
                            self.instance_buffer(obj_id, 0, matrices, version)
                        }
                    };
                    let idx = instance_buffers.len();
                    instance_buffers.push(buf);
                    // A `ShaderMaterial` reads the two custom columns too:
                    // eight floats an instance, zero-padded, in slot 2.
                    let custom_idx = items
                        .iter()
                        .any(|(m, _)| matches!(&**m, Material::Shader(_)))
                        .then(|| {
                            let b = match self.cached_instances(obj_id, 1, version) {
                                Some(b) => b,
                                None => {
                                    let mut data: Vec<f32> = Vec::with_capacity(im.transforms.len() * 8);
                                    for i in 0..im.transforms.len() {
                                        data.extend_from_slice(&im.custom_row(0, i));
                                        data.extend_from_slice(&im.custom_row(1, i));
                                    }
                                    self.instance_buffer(obj_id, 1, data, version)
                                }
                            };
                            instance_buffers.push(b);
                            instance_buffers.len() - 1
                        });
                    for (material, range) in &items {
                        let mut d = make_draw(&im.geometry, material, obj, Topology::Instanced);
                        d.range = *range;
                        d.instance_buf_idx = idx;
                        d.instance_stamp =
                            self.instance_stamp(obj_id, 0) ^ self.instance_stamp(obj_id, 1).rotate_left(32);
                        d.instance_count = im.transforms.len() as u32;
                        if let Some(((header, weights), tex)) = morph {
                            d.morph = header;
                            d.morph_weights = weights;
                            d.morph_tex = tex;
                        }
                        // Custom-shader material: its own pipeline, on the
                        // instanced layout — salted so it never shares a cache
                        // entry with the same material on a plain mesh.
                        if let (Material::Shader(sm), Some(custom_buf)) = (&**material, custom_idx) {
                            let hash = custom_shader_hash(&sm.fragment, sm.vertex.as_deref())
                                ^ INSTANCED_CUSTOM_SALT
                                ^ if sm.alpha_to_coverage { COVERAGE_CUSTOM_SALT } else { 0 };
                            let transparent = sm.transparent || sm.opacity < 1.0;
                            d.topology = Topology::Custom(hash, transparent, sm.screen_space, sm.side);
                            d.shader_idx = shader_data.len();
                            let pad = |s: &Vec<f32>| if s.is_empty() { vec![0.0f32] } else { s.clone() };
                            shader_data.push(ShaderDrawData {
                                fragment: sm.fragment.clone(),
                                vertex: sm.vertex.clone(),
                                data: sm.data_slots(),
                                storage: [pad(&sm.storage0), pad(&sm.storage1), pad(&sm.storage2)],
                                textures: sm.textures.clone(),
                            volumes: sm.volumes.clone(),
                                instanced: true,
                                custom_buf,
                                alpha_to_coverage: sm.alpha_to_coverage,
                            });
                        }
                        draws.push(d);
                    }
                }
                ObjectKind::SkinnedMesh(sm) => {
                    self.ensure_geometry(&sm.geometry);
                    let morph = crate::core::morph_uniforms(&sm.geometry, &sm.morph_influences)
                        .and_then(|u| Some((u, self.ensure_morph(&sm.geometry)?)));
                    let full = sm.geometry.draw_count() as u32;
                    let items: Vec<DrawItem<&std::sync::Arc<Material>>> = sm
                        .draws()
                        .into_iter()
                        .map(|i| (i.material, (i.start != 0 || i.end != full).then_some((i.start, i.end))))
                        .collect();
                    for &(material, range) in &items {
                        if material.wireframe() {
                            match range {
                                None => self.ensure_wire_indices(&sm.geometry),
                                Some((a, b)) => self.ensure_wire_span(&sm.geometry, a, b),
                            }
                        }
                    }
                    let with_span = |mut d: DrawMesh, range: Option<(u32, u32)>| -> DrawMesh {
                        d.range = range;
                        if let Some(((header, weights), tex)) = morph {
                            d.morph = header;
                            d.morph_weights = weights;
                            d.morph_tex = tex;
                        }
                        d
                    };
                    let key = geom_cache_key(&sm.geometry);
                    // Upload joints+weights to slot-1 vertex buffer (cached per-geometry).
                    if !self.skin_attr_cache.contains_key(&key) {
                        let joints = sm.geometry.get_attribute("joint");
                        let weights = sm.geometry.get_attribute("weight");
                        let (Some(j), Some(w)) = (joints, weights) else {
                            // Geometry isn't skin-equipped; draw at rest
                            // (morphed).
                            for &(material, range) in &items {
                                draws.push(with_span(
                                    make_draw(&sm.geometry, material, obj, Topology::Triangle),
                                    range,
                                ));
                            }
                            return;
                        };
                        let count = j.count();
                        let mut interleaved: Vec<f32> = Vec::with_capacity(count * 8);
                        for i in 0..count {
                            interleaved.extend_from_slice(&j.array[i * 4..i * 4 + 4]);
                            interleaved.extend_from_slice(&w.array[i * 4..i * 4 + 4]);
                        }
                        let buf = wgpu::util::DeviceExt::create_buffer_init(
                            &*self.device,
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("threers skin attrs"),
                                contents: bytemuck::cast_slice(&interleaved),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        );
                        self.skin_attr_cache.insert(key, buf);
                    }
                    // Upload bone matrices (per-frame, per-skinned-mesh).
                    //
                    // Always exactly `MAX_BONES` of them, padded with
                    // identities: the shader's array is fixed because WebGL2
                    // cannot do a runtime-sized one, and a uniform binding has
                    // to match the size the layout declares.
                    const MAX_BONES: usize = 128;
                    let mut bone_data: Vec<f32> = Vec::with_capacity(MAX_BONES * 16);
                    for m in sm.skeleton.bone_matrices.iter().take(MAX_BONES) {
                        bone_data.extend_from_slice(&m.elements);
                    }
                    let identity = crate::math::Matrix4::identity();
                    while bone_data.len() < MAX_BONES * 16 {
                        bone_data.extend_from_slice(&identity.elements);
                    }
                    let bone_buf = wgpu::util::DeviceExt::create_buffer_init(
                        &*self.device,
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("threers bone matrices"),
                            contents: bytemuck::cast_slice(&bone_data),
                            usage: wgpu::BufferUsages::UNIFORM,
                        },
                    );
                    // Defer creating the combined mesh+skin bind group until
                    // after per_mesh is populated below — we need the per-mesh
                    // uniform buffer at binding 0. One per draw: each group
                    // has its own mesh uniform, and the bones are shared.
                    let bone_hash = hash_words(bone_data.iter().map(|v| v.to_bits() as u64));
                    for &(material, range) in &items {
                        let idx = skin_data.len();
                        skin_data.push(SkinData {
                            bone_buf: bone_buf.clone(),
                            bg: None,
                            bone_hash,
                        });
                        let mut d = with_span(
                            make_draw(&sm.geometry, material, obj, Topology::Skinned),
                            range,
                        );
                        d.skin_idx = idx;
                        draws.push(d);
                    }
                }
                ObjectKind::Group => {}
            }
        });

        // Sort: opaque first (stable order), then transparents back-to-front so
        // alpha blending composites correctly. Sprites are already alpha-blended
        // and we treat them as transparent for ordering purposes.
        // Draw order class: opaque(0) → glass depth prepass(1) → glass color(2) →
        // ordinary alpha & OIT(3). Glass must run after opaque (so its prepass
        // sees opaque depth) and before ordinary alpha (which blends over it).
        fn ord_class(t: Topology) -> u8 {
            match t {
                Topology::GlassDepth => 1,
                Topology::GlassTint | Topology::GlassColor => 2,
                Topology::TriangleAlpha
                | Topology::PointAlpha
                | Topology::WideLineAlpha
                | Topology::Sprite
                | Topology::Oit
                | Topology::Refract
                | Topology::InstancedAlpha
                | Topology::SkinnedAlpha => 3,
                Topology::Custom(_, transparent, ..) if transparent => 3,
                _ => 0,
            }
        }
        // three.js renders the transmissive list between the opaque and the
        // transparent ones, so it can read the opaque scene behind it.
        let class = |d: &DrawMesh| -> u8 {
            match ord_class(d.topology) {
                0 if d.transmissive => 1,
                0 => 0,
                c => c + 1,
            }
        };
        // three.js draws a blended DoubleSide mesh twice — its back faces,
        // then its front faces (`WebGLRenderer.renderObject`, with the default
        // `forceSinglePass: false`) — so a translucent surface's far side always
        // lies under its near side. In one unculled pass the triangles blend
        // in index order instead, and a closed translucent shell shows its own
        // back faces over its front ones as triangle-shaped shards. The sort
        // below is stable and the pair shares every key, so it stays adjacent
        // and in this order. The back half casts no shadow: the front half's
        // record already covers a double-sided caster. Meshes, instanced and
        // skinned alike; a custom shader's pipeline is built apart from the cull
        // variants and still draws in one pass.
        let mut two_pass = Vec::with_capacity(draws.len());
        for d in draws.drain(..) {
            let blended_faces = matches!(
                d.topology,
                Topology::TriangleAlpha | Topology::InstancedAlpha | Topology::SkinnedAlpha
            );
            if blended_faces && d.side == 2 && !d.transmissive {
                let mut back = d.clone();
                back.side = 1;
                back.cast_shadow = false;
                let mut front = d;
                front.side = 0;
                two_pass.push(back);
                two_pass.push(front);
            } else {
                two_pass.push(d);
            }
        }
        draws = two_pass;
        // Opaque, then transmissive, then the translucent family (glass
        // prepass, glass colour, ordinary alpha). Within the family
        // `renderOrder` comes first, as it does in three.js's transparent list,
        // and only then the family's own order — so with equal render orders
        // every glass prepass still precedes every glass colour pass and both
        // precede ordinary alpha, while a glass shell given a higher order is
        // drawn after the translucent things inside it, whose colour it then
        // blends over instead of hiding them behind its prepass depth.
        let tier = |d: &DrawMesh| -> u8 { class(d).min(2) };
        draws.sort_by(|a, b| {
            tier(a)
                .cmp(&tier(b))
                .then(a.render_order.cmp(&b.render_order))
                .then(class(a).cmp(&class(b)))
                .then_with(|| {
                    // For ordinary transparents we want farther-from-camera first.
                    // view_z is negative in front of the camera; smaller = farther.
                    if ord_class(a.topology) == 3 && ord_class(b.topology) == 3 {
                        a.view_z
                            .partial_cmp(&b.view_z)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
        });

        // --- Frustum culling.
        //
        // Nothing here used to test whether a draw was on screen: every mesh in
        // the scene went through the vertex shader and got rasterised every
        // frame, however far outside the view it was. `Frustum` has been in
        // `src/math` the whole time, correct for this crate's 0..1 clip depth
        // and exposed to wasm, and the renderer simply never called it.
        //
        // Culled draws stay in `draws`. The shadow passes walk the same list
        // and a caster behind the camera can still throw a shadow right across
        // the view, so only the colour passes read this flag. The indices also
        // key `per_mesh` and the bind-group caches, and removing entries would
        // shift all of them.
        //
        // The jitter-free view-projection is deliberate: culling off the
        // jittered matrix would let an object on the boundary flicker in and
        // out from frame to frame.
        if self.frustum_culling {
            let frustum = Frustum::from_projection_matrix(&unjittered);
            for d in draws.iter_mut() {
                // Only geometry whose object-space bounds really do bound what
                // gets rasterised. A skinned mesh moves outside its bind pose,
                // an instanced draw covers ground the source geometry does not,
                // a custom vertex shader may put the vertices anywhere, and the
                // sky is deliberately pinned to the far plane.
                let plain = matches!(
                    d.topology,
                    Topology::Triangle
                        | Topology::TriangleAlpha
                        | Topology::TriangleNoCull
                        | Topology::TriangleWire
                        | Topology::Line
                        | Topology::WideLine
                        | Topology::WideLineAlpha
                        | Topology::GlassDepth
                        | Topology::GlassTint
                        | Topology::GlassColor
                        | Topology::Oit
                        | Topology::Refract
                ) && d.instance_count == 1
                    && d.skin_idx == usize::MAX
                    && d.shader_idx == usize::MAX
                    // A morphed mesh can leave its rest-pose bounds.
                    && d.morph[0] == 0.0;
                if !plain {
                    continue;
                }
                let Some(sphere) = self.geom_cache.get(&d.key).and_then(|g| g.sphere) else {
                    continue;
                };
                let world = sphere.apply_matrix4(&Matrix4 { elements: d.model });
                d.in_frustum = frustum.intersects_sphere(&world);
            }
        }
        // The transmission capture passes (`capture_transmission`): the first
        // draws everything but the transmissive objects, the second only the
        // back faces of the double-sided ones, as three.js's
        // `renderTransmissionPass` does. Through `in_frustum`, so the shadow
        // passes still see every caster.
        if let Some(back) = self.capturing_transmission {
            for d in draws.iter_mut() {
                let keep = if back { d.transmissive && d.side == 2 } else { !d.transmissive };
                if !keep {
                    d.in_frustum = false;
                } else if back {
                    d.side = 1;
                }
            }
        }
        self.cull_stats = (
            draws.iter().filter(|d| d.in_frustum).count(),
            draws.iter().filter(|d| !d.in_frustum).count(),
        );

        frame_u.ambient = [ambient[0], ambient[1], ambient[2], 0.0];
        frame_u.light_counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        // w: the bound capture's mip count — the stand-in's one while a capture
        // pass draws, the capture's after it.
        let capture_levels = match (self.capturing_transmission, &self.transmission) {
            (Some(false), _) | (None, None) => 1,
            (Some(true), Some(c)) | (None, Some(c)) => c.levels,
            (Some(true), None) => 1,
        };
        frame_u.light_counts2 = [
            n_rect as u32,
            self.specular_antialiasing as u32,
            self.reflective_transparency as u32,
            capture_levels,
        ];

        // Each shadow-casting light's record — layers, matrices, biases and
        // filter — and the passes that draw its maps. A directional light's
        // single map fits a fixed cube on the world origin
        // (`ShadowSettings::camera_size`), so it depends on lights and
        // casters, never on the camera; cascades follow the camera.
        //
        // The maps are allocated before any light exists to ask for a size,
        // so the first frame that names one reallocates. `map_size` used to be
        // read and discarded, which is a worse failure than not offering it:
        // the caller sets 4096, sees no error, and gets 1024.
        let layers_of = |kind: u8| -> usize {
            casters
                .iter()
                .filter(|c| matches!((kind, c), (0, Caster::Dir { .. }) | (1, Caster::Spot { .. }) | (2, Caster::Point { .. })))
                .map(Caster::layers)
                .sum()
        };
        let (dir_layers, spot_layers, point_layers) = (layers_of(0), layers_of(1), layers_of(2));
        self.ensure_shadow_arrays(
            casters
                .iter()
                .filter_map(|c| match c {
                    Caster::Dir { settings, .. } => Some(settings.map_size),
                    _ => None,
                })
                .max()
                .unwrap_or(0),
            dir_layers,
            spot_layers,
            point_layers / 6,
        );
        let mut shadow_passes: Vec<ShadowPass> = Vec::new();
        let mut next_layer = [0usize; 3];
        for (r, caster) in casters.iter().enumerate() {
            let settings = caster.settings();
            let filter = settings.filter.unwrap_or(self.shadow_type).code();
            let bias = settings.bias.max(0.0);
            let normal_bias = settings.normal_bias.max(0.0);
            let mut rec = ShadowRecordGpu::zeroed();
            match *caster {
                Caster::Dir { dir, settings } => {
                    let size = self.dir_shadows.size;
                    let texel = 1.0 / size as f32;
                    let layer = next_layer[0];
                    let cascades = caster.layers();
                    next_layer[0] += cascades;
                    // The light's size as a penumbra per unit of normalised
                    // depth, in map uv: an angle times the depth range over
                    // the map's width. Cascades keep that ratio fixed (see
                    // `cascade_frusta`), so one number serves them all.
                    let angle = if settings.light_size > 0.0 { settings.light_size } else { 0.02 };
                    let (depth_range, width) = if cascades > 1 {
                        (12.0, 2.0)
                    } else {
                        (settings.camera_far - settings.camera_near, 2.0 * settings.camera_size)
                    };
                    let scale = angle * depth_range / width.max(1e-6);
                    rec.info = [0, layer as u32, cascades as u32, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.25).clamp(2.0 * texel, 0.1), 0.0, 0.0, scale];
                    if cascades > 1 {
                        for (c, (vp, far)) in cascade_frusta(&view_m, &camera.projection_matrix(), dir, &settings, size).into_iter().enumerate() {
                            rec.vp[c] = vp;
                            rec.extra[c] = far;
                            shadow_passes.push(ShadowPass { kind: ShadowKind::Dir, layer: (layer + c) as u32, vp, pos: [0.0; 4] });
                        }
                    } else {
                        let s = settings.camera_size;
                        let light_eye = -dir.normalize() * (s * 1.5);
                        let light_view = Matrix4::look_at(light_eye, Vector3::ZERO, Vector3::UP);
                        let light_proj =
                            Matrix4::orthographic(-s, s, s, -s, settings.camera_near, settings.camera_far);
                        let vp = light_proj.multiply(&light_view).elements;
                        rec.vp[0] = vp;
                        rec.extra = [f32::MAX; 4];
                        shadow_passes.push(ShadowPass { kind: ShadowKind::Dir, layer: layer as u32, vp, pos: [0.0; 4] });
                    }
                }
                Caster::Spot { pos, dir, angle, settings } => {
                    let texel = 1.0 / self.spot_shadows.size as f32;
                    let layer = next_layer[1];
                    next_layer[1] += 1;
                    let target = pos + dir.normalize();
                    let light_view = Matrix4::look_at(pos, target, Vector3::UP);
                    // Perspective frustum: fov = 2 * outer_angle to cover the cone.
                    let fov = (angle * 2.0).clamp(0.05, std::f32::consts::PI - 0.05);
                    let light_proj =
                        Matrix4::perspective(fov, 1.0, settings.camera_near, settings.camera_far);
                    let vp = light_proj.multiply(&light_view).elements;
                    // The light's width over the frustum's width at unit
                    // distance: the penumbra in uv per unit of (receiver −
                    // blocker) / (blocker · receiver).
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    let scale = size / (2.0 * (fov * 0.5).tan());
                    rec.info = [1, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.05).clamp(2.0 * texel, 0.1), settings.camera_near, settings.camera_far, scale];
                    rec.vp[0] = vp;
                    shadow_passes.push(ShadowPass { kind: ShadowKind::Spot, layer: layer as u32, vp, pos: [0.0; 4] });
                }
                Caster::Point { pos, radius, settings } => {
                    // The filter disc lies across the direction to the light,
                    // a face texel (`2 / size` of a unit cube face) per unit
                    // of radius.
                    let texel = 2.0 / POINT_SHADOW_SIZE as f32;
                    let layer = next_layer[2];
                    next_layer[2] += 6;
                    let at = [pos.x, pos.y, pos.z, radius];
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    rec.info = [2, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.extra = at;
                    rec.pcss = [(size * 0.05).clamp(2.0 * texel, 0.2), 0.0, 0.0, size];
                    for face in 0..6 {
                        shadow_passes.push(ShadowPass {
                            kind: ShadowKind::Point,
                            layer: (layer + face) as u32,
                            vp: point_face_vp(pos, radius, face),
                            pos: at,
                        });
                    }
                }
            }
            frame_u.shadows[r] = rec;
        }
        frame_u.shadow_info = [casters.len() as u32, 0, 0, 0];

        // Sync environment cubemap. Upload on first sight, rebuild bind group on change.
        // A cube texture is prefiltered first, as three.js's `WebGLCubeUVMaps`
        // does for every Standard/Physical environment.
        let scene_env_cube = scene.environment.as_ref().map(|e| self.prefiltered(e));
        let env_key = scene_env_cube.as_ref().map(cube_cache_key);
        let env_rt_id = scene.environment_cube_rt;
        if self.env_dirty || env_key != self.env_cached_key || env_rt_id != self.env_cached_rt_id {
            let mut env_map_params = [0.0_f32; 4];
            if let Some(env) = &scene_env_cube {
                let k = cube_cache_key(env);
                if !self.cube_cache.contains_key(&k) {
                    let gc = GpuCubeTexture::upload(&self.device, &self.queue, env);
                    self.cube_cache.insert(k, gc);
                }
                if let Some(atlas) = &env.cube_uv_atlas {
                    env_map_params = [
                        atlas.texel_width,
                        atlas.texel_height,
                        atlas.lod_max as f32,
                        1.0,
                    ];
                }
            }
            frame_u.env_map_params = env_map_params;
            let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene_env_cube {
                self.cube_cache
                    .get(&cube_cache_key(env))
                    .and_then(|gc| gc.cube_uv_view.as_ref())
                    .unwrap_or(&self.default_cube_uv_view)
            } else {
                &self.default_cube_uv_view
            };
            let view: &wgpu::TextureView = if let Some(rt_id) = scene.environment_cube_rt {
                self.cube_rt_view_cache
                    .get(&rt_id)
                    .unwrap_or(&self.default_env_cube.view)
            } else if let Some(env) = &scene_env_cube {
                &self.cube_cache[&cube_cache_key(env)].view
            } else {
                &self.default_env_cube.view
            };
            // Whatever `view` just became, record how many mips the shader may
            // sample from it. A cube render target's view is built with
            // `mip_level_count: Some(1)` and the placeholder cube is 1x1, so
            // both expose a single level; only an uploaded environment has a
            // chain worth walking.
            self.env_max_mip = if scene.environment_cube_rt.is_some() {
                0.0
            } else if let Some(env) = &scene_env_cube {
                let levels = self.cube_cache[&cube_cache_key(env)]
                    .texture
                    .mip_level_count();
                levels.saturating_sub(1) as f32
            } else {
                0.0
            };
            self.env_bind_group = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers env bg",
                view,
                &self.env_sampler,
                &self.dir_shadows.view,
                &self.shadow_sampler,
                &self.spot_shadows.view,
                &self.spot_shadow_sampler,
                &self.point_shadows.view,
                &self.point_shadow_sampler,
                cube_uv_view,
                &self.env_uniform,
                &self.ltc.1,
            );
            if let Some(bgl) = &self.custom_env_bgl {
                // The trimmed custom environment follows the same cube and 2D
                // fallback; its uniform is the shared `env_uniform` buffer.
                self.custom_env_bind_group = Some(custom_env_bind_group!(
                    self.device,
                    bgl,
                    "threers custom env bg",
                    view,
                    &self.env_sampler,
                    cube_uv_view,
                    &self.env_uniform,
                ));
            }
            self.env_dirty = false;
            // Theirs share the shadow-map views just rebound.
            self.material_env_groups.clear();
            self.env_cached_key = env_key;
            self.env_cached_rt_id = env_rt_id;
        }
        // The plain-cube path reads its chain length from here as well, so this
        // has to be set on every frame: the bind-group block above only runs
        // when the environment changes, and `frame_u` starts from `Default`.
        frame_u.env_map_params[2] = self.env_max_mip;
        if let Some(env) = &scene_env_cube {
            if let Some(atlas) = &env.cube_uv_atlas {
                frame_u.env_map_params = [
                    atlas.texel_width,
                    atlas.texel_height,
                    atlas.lod_max as f32,
                    1.0,
                ];
            }
        }
        // The scene's environment as group 3 reads it: three.js's
        // `scene.environmentIntensity` and `environmentRotation`.
        let scene_env = EnvUniforms {
            params: frame_u.env_map_params,
            info: [has_env as u32 as f32, scene.environment_intensity, 0.0, 0.0],
            rot: crate::materials::environment_lookup_rows(
                scene.environment_rotation,
                scene.environment_cube_rt.is_none()
                    && scene_env_cube.as_ref().is_some_and(|e| e.cube_uv_atlas.is_none()),
            ),
        };
        self.queue.write_buffer(&self.env_uniform, 0, bytemuck::bytes_of(&scene_env));

        // -- Ensure every texture referenced in this frame is uploaded. --
        // Every role's map — the list `material_maps` binds from, so a new
        // role cannot be bound without being uploaded.
        for d in &draws {
            for (_, t) in super::material_maps::role_textures(&d.slots) {
                self.ensure_slot_uploaded(&t.cloned());
            }
            // A Standard/Physical material's own `envMap` prefiltered, as the
            // scene's is; a Basic/Lambert/Phong one sampled as it is.
            let pbr = d.slots.pbr_env_map.as_ref().map(|c| self.prefiltered(c));
            for cube in [d.slots.env_map.as_ref(), pbr.as_ref()].into_iter().flatten() {
                let k = cube_cache_key(cube);
                if !self.cube_cache.contains_key(&k) {
                    let gc = GpuCubeTexture::upload(&self.device, &self.queue, cube);
                    self.cube_cache.insert(k, gc);
                }
            }
        }
        // Group 3 for each Standard/Physical material's own `envMap`.
        for d in &draws {
            let (Some(key), Some(source)) = (d.env_group, d.slots.pbr_env_map.as_ref()) else {
                continue;
            };
            if self.material_env_groups.contains_key(&key) {
                continue;
            }
            let cube = &self.prefiltered(source);
            let gc = &self.cube_cache[&cube_cache_key(cube)];
            let u = EnvUniforms::of(cube, gc, f32::from_bits(key.1), crate::math::Euler {
                x: f32::from_bits(key.2[0]),
                y: f32::from_bits(key.2[1]),
                z: f32::from_bits(key.2[2]),
                order: crate::math::EulerOrder::ALL[key.2[3] as usize],
            });
            let buf = wgpu::util::DeviceExt::create_buffer_init(
                &*self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("threers material env uniforms"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let bg = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers material env bg",
                &gc.view,
                &self.env_sampler,
                &self.dir_shadows.view,
                &self.shadow_sampler,
                &self.spot_shadows.view,
                &self.spot_shadow_sampler,
                &self.point_shadows.view,
                &self.point_shadow_sampler,
                gc.cube_uv_view.as_ref().unwrap_or(&self.default_cube_uv_view),
                &buf,
                &self.ltc.1,
            );
            self.material_env_groups.insert(key, (buf, bg));
        }

        // -- Ensure pool capacity: the uniform arena, and a mesh bind group
        // over each draw's slice of it. Grown by half again, so a scene
        // gaining meshes one at a time does not rebuild it every frame. --
        let stride = self.mesh_stride;
        let mesh_size = wgpu::BufferSize::new(std::mem::size_of::<MeshUniforms>() as u64);
        let capacity = self.mesh_arena.as_ref().map_or(0, |b| (b.size() / stride) as usize);
        if capacity < draws.len() {
            let capacity = (draws.len() + draws.len() / 2).max(64);
            let arena = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers mesh uniforms"),
                size: capacity as u64 * stride,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = |i: usize| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers mesh bg"),
                    layout: &self.mesh_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena,
                            offset: i as u64 * stride,
                            size: mesh_size,
                        }),
                    }],
                })
            };
            let rebuilt: Vec<wgpu::BindGroup> = (0..capacity).map(bind).collect();
            for (i, mesh_bg) in rebuilt.into_iter().enumerate() {
                if let Some(entry) = self.per_mesh.get_mut(i) {
                    entry.0 = i as u64 * stride;
                    entry.1 = mesh_bg;
                } else {
                    // Texture bind group is rebuilt per-frame per-mesh below,
                    // but the pool entry needs a placeholder until then.
                    let tex_bg = std::sync::Arc::new(self.make_default_tex_bg());
                    self.per_mesh.push((i as u64 * stride, mesh_bg, tex_bg));
                }
            }
            self.mesh_arena = Some(arena);
        }
        let mut mesh_bytes = std::mem::take(&mut self.mesh_bytes);
        mesh_bytes.clear();
        mesh_bytes.resize(draws.len() * stride as usize, 0);

        // -- Build combined mesh+skin bind groups now that per_mesh is grown. --
        for (i, d) in draws.iter().enumerate() {
            if d.topology.is_skinned() && d.skin_idx < skin_data.len() {
                let sd = &mut skin_data[d.skin_idx];
                let arena = self.mesh_arena.as_ref().expect("grown above");
                sd.bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers mesh skinned bg"),
                    layout: &self.mesh_skinned_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: arena,
                                offset: self.per_mesh[i].0,
                                size: mesh_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sd.bone_buf.as_entire_binding(),
                        },
                    ],
                }));
            }
        }

        // The topologies the multisampled pipelines exist for.
        let msaa_drawable = |t: Topology| {
            matches!(
                t,
                Topology::Triangle
                    | Topology::TriangleAlpha
                    | Topology::TriangleNoCull
                    | Topology::Sky
                    | Topology::TriangleWire
                    | Topology::Line
                    | Topology::Point
                    | Topology::WideLine
                    | Topology::GlassDepth
                    | Topology::GlassTint
                    | Topology::GlassColor
                    | Topology::Skinned
                    | Topology::Instanced
                    // An opaque shader material (see `custom_msaa_pipelines`).
                    | Topology::Custom(_, false, false, _)
            )
        };
        // Multisampling in a pane with a partial clear: the pane is drawn into
        // a target of its own, the size of the clear rectangle, and
        // composited into this one — see `viewport::PanePlan`. Its viewport is
        // relative to that target, so the projection is redone for it here,
        // before the frame uniforms go up.
        let pane_msaa = vp_plan.pane.filter(|_| {
            self.msaa > 1
                && !self.cube_pass
                && depth_src.is_none()
                && !linear_framebuffer
                && self.jitter_override.is_none()
                && draws.iter().all(|d| msaa_drawable(d.topology))
        });
        if let Some(p) = &pane_msaa {
            let mut proj = camera.projection_matrix();
            if let Some(fix) = &p.projection_fix {
                proj = fix.multiply(&proj);
            }
            if self.state_pipelines.reversed() {
                proj = reverse_depth(&proj);
            }
            frame_u.projection = proj.elements;
            frame_u.view_proj = proj.multiply(&view_m).elements;
        }

        // Alpha blending as three.js's default renderer does it — on the
        // sRGB-encoded bytes of its 8-bit canvas, not in linear light. A plain
        // `Unorm` target already works that way (the shader encodes). An sRGB
        // one blends in linear in hardware, so its blended mesh draws go in a
        // pass of their own, after the rest, through a `Unorm` view of the same
        // texture with the shader encoding — the hardware then mixes encoded
        // values exactly as WebGL does, while the opaque pass, its multisample
        // resolve and everything downstream stay in linear light. Needs the
        // target texture (the `render_to` paths hand it over) and view-format
        // reinterpretation (not on WebGL2); elsewhere the blend stays linear.
        //
        // Every blended kind goes there (`Topology::encodes`): meshes,
        // sprites, transparent instanced and skinned meshes and transparent
        // custom shaders. In a pane drawn through its own multisampled target
        // the pass draws into that target's `Unorm` view; in a frame with
        // screen-space refraction, into the opaque capture's, which is then
        // blitted to the target as before.
        let encoded_target: Option<&wgpu::Texture> = taa_src
            .map(|(_, c)| c)
            .filter(|t| t.format().is_srgb() && t.format() == self.color_format)
            .filter(|t| t.sample_count() == 1 && t.mip_level_count() == 1);
        //
        // Without view reinterpretation (WebGL2) the pass draws into a `Unorm`
        // scratch copy of the target instead, copied back after
        // (`ensure_encoded_scratch`) — for a single-sampled frame without a
        // pane target or screen-space refraction, which is what a WebGL2
        // canvas drawn through its back buffer is.
        let reinterpret = super::caps::supports_view_formats(&self.device);
        let scratch_ok = !reinterpret
            && self.msaa <= 1
            && pane_msaa.is_none()
            && !draws.iter().any(|d| {
                d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, ..))
            })
            && encoded_target.is_some_and(|t| t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING));
        let encoded_pass = encoded_target.is_some()
            && !linear_framebuffer
            && !self.cube_pass
            && self.encoded_blend_pipelines.is_some()
            && (reinterpret || scratch_ok)
            && draws.iter().any(|d| d.topology.encodes());
        let encoded_scratch = encoded_pass && !reinterpret;
        if let (true, Some(t)) = (encoded_scratch, encoded_target) {
            self.ensure_encoded_scratch(t.width(), t.height());
        }

        // Material shader variants: a draw whose maps the legacy layout cannot
        // hold (or whose maps disagree on filtering or transform) gets one,
        // built here, before any pass needs its pipelines. Custom shader
        // materials compile against the legacy layout and keep it.
        for d in draws.iter_mut() {
            d.map_variant = (!matches!(d.topology, Topology::Custom(..))
                && super::material_maps::needs_variant(&d.slots))
            .then(|| super::material_maps::MapVariant::of(&d.slots));
        }
        let wanted: std::collections::HashSet<super::material_maps::MapVariant> =
            draws.iter().filter_map(|d| d.map_variant).collect();
        for v in wanted {
            self.ensure_map_variant(v);
        }

        // -- Upload uniforms + rebuild per-mesh tex bind groups. --
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame_u));
        for (i, d) in draws.iter().enumerate() {
            let mut flags: u32 = 0;
            if d.slots.map.is_some() {
                flags |= FLAG_MAP;
            }
            if d.slots.normal_map.is_some() {
                flags |= FLAG_NORMAL_MAP;
            }
            if d.slots.roughness_map.is_some() {
                flags |= FLAG_ROUGHNESS_MAP;
            }
            if d.slots.metalness_map.is_some() {
                flags |= FLAG_METALNESS_MAP;
            }
            if d.slots.ao_map.is_some() {
                flags |= FLAG_AO_MAP;
            }
            if d.slots.emissive_map.is_some() {
                flags |= FLAG_EMISSIVE_MAP;
            }
            if d.slots.matcap_map.is_some() {
                flags |= FLAG_MATCAP_MAP;
            }
            if d.slots.cloud_shadow_map.is_some() && d.atmos[1] > 0.0 {
                flags |= FLAG_CLOUD_SHADOW;
            }

            let u = MeshUniforms {
                model: d.model,
                normal_matrix: d.normal_matrix,
                color: d.color,
                emissive: d.emissive,
                specular: d.specular,
                params: [
                    d.shininess_or_near,
                    d.opacity_or_far,
                    d.roughness,
                    d.metalness,
                ],
                params2: [
                    d.ao_intensity,
                    d.normal_scale[0],
                    d.normal_scale[1],
                    0.0,
                ],
                params3: d.physical,
                params4: d.physical2,
                params5: d.physical3,
                params6: d.physical4,
                params7: d.physical5,
                params8: d.displacement,
                // A variant transforms each map's uv itself; its vertex stage
                // passes the raw uv through.
                params9: if d.map_variant.is_some() { [0.0, 0.0, 1.0, 1.0] } else { d.uv_transform },
                params10: d.atmos,
                params11: d.twilight,
                params12: d.eclipse,
                params13: d.scatter,
                params14: d.aurora,
                flags: [
                    d.kind as u32,
                    flags,
                    d.shader_flags
                        | if encoded_pass && d.topology.encodes() {
                            FLAG_ENCODED_OUT
                        } else {
                            0
                        },
                    d.alpha_test.to_bits(),
                ],
                // `morph.w` bit 32: the geometry has extra uv sets.
                morph: {
                    let mut m = d.morph;
                    if self.uv_cache.contains_key(&d.key) {
                        m[3] = (m[3] as u32 | 32) as f32;
                    }
                    m
                },
                morph_weights: bytemuck::cast(d.morph_weights),
                clip_plane: d.clip_plane,
            };
            let o = self.per_mesh[i].0 as usize;
            mesh_bytes[o..o + std::mem::size_of::<MeshUniforms>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        if let (Some(arena), false) = (&self.mesh_arena, mesh_bytes.is_empty()) {
            self.queue.write_buffer(arena, 0, &mesh_bytes);
        }
        self.mesh_bytes = mesh_bytes;
        // Texture bind groups, from the cache. Keys are computed up front so
        // the miss path can borrow `self` immutably to build and then mutably
        // to insert, one draw at a time.
        let tex_keys: Vec<TexBgKey> = draws
            .iter()
            .map(|d| {
                let (maps, _, sampler, _) = tex_bg_key(&d.slots);
                (maps, d.morph_tex, sampler, self.uv_key(d.key))
            })
            .collect();
        // The set of distinct texture combinations in a scene is small and
        // static; this bound is a runaway guard, not a working limit.
        if self.tex_bg_cache.len() > 4096 {
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
        for (i, d) in draws.iter().enumerate() {
            if !self.tex_bg_cache.contains_key(&tex_keys[i]) {
                let morph = self
                    .morph_cache
                    .get(&(d.morph_tex.0 as *const BufferGeometry))
                    .filter(|_| d.morph_tex.0 != 0)
                    .map(|m| &m.view);
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                let bg = std::sync::Arc::new(self.build_tex_bg(&d.slots, morph, uv));
                self.tex_bg_cache.insert(tex_keys[i], bg);
            }
        }
        for (i, k) in tex_keys.iter().enumerate() {
            self.per_mesh[i].2 = self.tex_bg_cache[k].clone();
        }
        // Variant draws: their own group 2.
        if self.variant_bg_cache.len() > 4096 {
            self.variant_bg_cache.clear();
        }
        for (i, d) in draws.iter().enumerate() {
            let Some(v) = d.map_variant else { continue };
            let key = (variant_bg_key(&d.slots, v, d.morph_tex, d.map_uniforms), self.uv_key(d.key));
            if !self.variant_bg_cache.contains_key(&key) {
                let morph = self
                    .morph_cache
                    .get(&(d.morph_tex.0 as *const BufferGeometry))
                    .filter(|_| d.morph_tex.0 != 0)
                    .map(|m| &m.view);
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                let bg = std::sync::Arc::new(self.build_variant_tex_bg(&d.slots, morph, uv, v, d.map_uniforms));
                self.variant_bg_cache.insert(key.clone(), bg);
            }
            self.per_mesh[i].2 = self.variant_bg_cache[&key].clone();
        }
        // A shader material's group 2 when its geometry is morphed or has
        // extra uv sets: the shared custom one with the geometry's morph
        // texture and uv sets in place of the empty defaults. (Its `morph.w`
        // bit 32 is set whenever the uv sets exist, so the shared group's
        // empty texture would be read as zeros.)
        let custom_morph_bgs: Vec<Option<wgpu::BindGroup>> = draws
            .iter()
            .map(|d| {
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                if !matches!(d.topology, Topology::Custom(..)) || (d.morph_tex.0 == 0 && uv.is_none()) {
                    return None;
                }
                let view = if d.morph_tex.0 == 0 {
                    &self.default_morph.1
                } else {
                    &self.morph_cache.get(&(d.morph_tex.0 as *const BufferGeometry))?.view
                };
                let uv = uv.unwrap_or(&self.default_uv_sets.1);
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers custom morph tex bg"),
                    layout: &self.custom_tex_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(uv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&self.sampler_linear_repeat),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                    ],
                }))
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers encoder"),
            });

        // Whether this render redraws the shadow maps: always, unless
        // `shadow_map.auto_update` is off — then only when asked, or when the
        // maps were last drawn for other casters or another size. They
        // depend on lights and casters, never on the camera (the directional
        // map is a fixed cube on the origin), so a skipped update samples the
        // same maps a redraw would have produced.
        //
        // With `auto_update` on (the default), a render whose casters and
        // lights are the same as last time keeps the maps it has: a still
        // scene under a moving camera stops redrawing every caster into every
        // map each frame.
        let mut pass_hashes: Vec<Option<u64>> = Vec::new();
        let mut pass_changed: Vec<bool> = Vec::new();
        let update_shadows = !shadow_passes.is_empty() && {
            let sig: ShadowSig = (dir_layers, spot_layers, point_layers, self.dir_shadows.size, self.spot_shadows.size);
            let valid = self.shadow_sig == Some(sig);
            let content = {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut mix = |x: u64| h = mix_word(h, x);
                let mut known = true;
                for d in draws.iter().filter(|d| d.cast_shadow) {
                    // Skinned: the pose. Instanced: the instance data, through
                    // its content stamp.
                    if d.topology.is_skinned() {
                        mix(skin_data.get(d.skin_idx).map_or(0, |s| s.bone_hash));
                    }
                    mix(d.instance_stamp);
                    mix(d.instance_count as u64);
                    // A custom shader is a function of its inputs — there is
                    // no clock in the frame uniforms — so its data, storage and
                    // source stand for its output. Not its textures: a texture
                    // can be drawn into without anything here changing.
                    if let Topology::Custom(..) = d.topology {
                        match shader_data.get(d.shader_idx) {
                            Some(sd) if sd.textures.is_empty() && sd.volumes.is_empty() => {
                                sd.data.iter().flatten().for_each(|v| mix(v.to_bits() as u64));
                                sd.storage.iter().for_each(|st| {
                                    mix(st.len() as u64);
                                    st.iter().for_each(|v| mix(v.to_bits() as u64));
                                });
                                mix(hash_words(sd.fragment.bytes().map(u64::from)));
                                mix(sd.vertex.as_deref().map_or(0, |v| hash_words(v.bytes().map(u64::from))));
                            }
                            _ => {
                                known = false;
                                break;
                            }
                        }
                    }
                    mix(d.key as usize as u64);
                    mix(self.geom_cache.get(&d.key).map_or(0, |e| e.version) as u64);
                    d.model.iter().for_each(|v| mix(v.to_bits() as u64));
                    mix(d.shadow_side as u64);
                    mix(d.alpha_test.to_bits() as u64);
                    mix(d.slots.map.as_ref().map_or(0, |t| std::sync::Arc::as_ptr(t) as usize as u64));
                    mix(d.slots.alpha_map.as_ref().map_or(0, |t| std::sync::Arc::as_ptr(t) as usize as u64));
                    mix(d.shader_flags as u64);
                    d.color.iter().chain(&d.physical).chain(&d.morph).for_each(|v| mix(v.to_bits() as u64));
                    d.morph_weights.iter().for_each(|v| mix(v.to_bits() as u64));
                    mix(d.range.map_or(u64::MAX, |(a, b)| ((a as u64) << 32) | b as u64));
                }
                known.then_some(h)
            };
            // Each pass's own: the casters and its light's matrix. A pass is
            // redrawn when its own changed — a cascade following the camera
            // does not take a still point light's six faces with it.
            pass_hashes = shadow_passes
                .iter()
                .map(|p| {
                    content.map(|c| {
                        let mut h = mix_word(c, p.layer as u64 | ((p.kind as u64) << 32));
                        for v in p.vp.iter().chain(&p.pos) {
                            h = mix_word(h, v.to_bits() as u64);
                        }
                        h
                    })
                })
                .collect();
            pass_changed = pass_hashes
                .iter()
                .enumerate()
                .map(|(k, h)| !valid || h.is_none() || self.shadow_pass_hashes.get(k) != Some(h))
                .collect();
            let unchanged = !pass_changed.iter().any(|&c| c);
            self.shadow_sig = Some(sig);
            self.shadow_map.take_update(valid, unchanged)
        };
        // A forced update (`needs_update` with `auto_update` off) redraws
        // every pass; an automatic one, the passes that changed.
        let draw_pass: Vec<bool> = (0..shadow_passes.len())
            .map(|k| update_shadows && (!self.shadow_map.auto_update || pass_changed.get(k).copied().unwrap_or(true)))
            .collect();
        if update_shadows {
            self.shadow_pass_hashes = std::mem::take(&mut pass_hashes);
        }
        let update_shadows = draw_pass.iter().any(|&d| d);
        if update_shadows {
            self.shadow_updates += 1;
            // Every pass's matrix, at its own aligned offset in one buffer,
            // written once.
            self.ensure_shadow_pass_buffer(shadow_passes.len());
            let stride = self.shadow_pass_stride as usize;
            let mut bytes = vec![0u8; stride * shadow_passes.len()];
            for (k, p) in shadow_passes.iter().enumerate() {
                let g = ShadowPassGpu { vp: p.vp, light: p.pos };
                bytes[k * stride..k * stride + std::mem::size_of::<ShadowPassGpu>()]
                    .copy_from_slice(bytemuck::bytes_of(&g));
            }
            if let Some((buffer, _)) = &self.shadow_pass_buffer {
                self.queue.write_buffer(buffer, 0, &bytes);
            }
        }

        // The shadow depth passes: every caster from each light, one layer
        // each (six for a point light). The timer brackets the first pass's
        // start and the last one's end.
        // Each caster's world bounding sphere, where its geometry's bounds
        // really bound what it draws (as the main pass culls): a pass skips
        // casters outside its light's frustum.
        let caster_spheres: Vec<Option<crate::math::Sphere>> = draws
            .iter()
            .map(|d| {
                let plain = d.cast_shadow
                    && (matches!(d.topology, Topology::Triangle | Topology::TriangleNoCull | Topology::TriangleAlpha)
                        || d.topology.is_glass_mesh())
                    && d.instance_count == 1
                    && d.skin_idx == usize::MAX
                    && d.shader_idx == usize::MAX
                    && d.morph[0] == 0.0;
                if !plain {
                    return None;
                }
                let sphere = self.geom_cache.get(&d.key).and_then(|g| g.sphere)?;
                Some(sphere.apply_matrix4(&Matrix4 { elements: d.model }))
            })
            .collect();
        let drawn: Vec<usize> = (0..shadow_passes.len()).filter(|&k| draw_pass[k]).collect();
        self.shadow_stats.0 += drawn.len() as u64;
        for (n, &k) in drawn.iter().enumerate() {
            let p = &shadow_passes[k];
            let pass_frustum = Frustum::from_projection_matrix(&Matrix4 { elements: p.vp });
            let view = match p.kind {
                ShadowKind::Dir => self.dir_shadows.layer_view(p.layer),
                ShadowKind::Spot => self.spot_shadows.layer_view(p.layer),
                ShadowKind::Point => self.point_shadows.layer_view(p.layer),
            };
            let timestamp_writes = self.shadow_timer.as_ref().and_then(|t| match (n == 0, n + 1 == drawn.len()) {
                (true, true) => Some(t.mark_both(0, 1)),
                (true, false) => Some(t.mark(0)),
                (false, true) => Some(t.mark_end(1)),
                _ => None,
            });
            let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(match p.kind {
                    ShadowKind::Dir => "threers shadow pass",
                    ShadowKind::Spot => "threers spot shadow pass",
                    ShadowKind::Point => "threers point shadow face pass",
                }),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let Some((_, pass_bg)) = &self.shadow_pass_buffer else { break };
            spass.set_bind_group(0, pass_bg, &[(k as u64 * self.shadow_pass_stride) as u32]);
            for (i, d) in draws.iter().enumerate() {
                // three.js draws only `castShadow` objects into a shadow map.
                if !d.cast_shadow {
                    continue;
                }
                if caster_spheres[i].as_ref().is_some_and(|s| !pass_frustum.intersects_sphere(s)) {
                    continue;
                }
                self.shadow_stats.1 += 1;
                let skinned = matches!(d.topology, Topology::Skinned | Topology::SkinnedAlpha);
                let instanced = matches!(d.topology, Topology::Instanced | Topology::InstancedAlpha);
                // A transparent caster casts too: three.js's shadow map ignores
                // blending (only `alphaTest` cuts a shadow). So does glass in
                // the modes with passes of their own: solid glass fully, thin
                // glass through the cut-out pipelines (`shadow_transmits`).
                if d.topology != Topology::Triangle
                    && d.topology != Topology::TriangleNoCull
                    && d.topology != Topology::TriangleAlpha
                    && !d.topology.is_glass_mesh()
                    && !skinned
                    && !instanced
                {
                    continue;
                }
                let gm = &self.geom_cache[&d.key].mesh;
                let mut instances = 0..1;
                // three.js's shadow depth material takes `map` and `alphaTest`;
                // the point passes have no cut-out variant.
                let cut = p.kind != ShadowKind::Point
                    && (d.shadow_transmits
                        || (d.alpha_test > 0.0 && (d.slots.map.is_some() || d.slots.alpha_map.is_some())));
                let c = &self.shadow_cut;
                let base = match (p.kind, skinned, instanced, cut) {
                    (ShadowKind::Dir, true, _, false) => &self.pipeline_shadow_skinned,
                    (ShadowKind::Dir, true, _, true) => &c.dir_skinned,
                    (ShadowKind::Dir, false, true, false) => &self.pipeline_shadow_instanced,
                    (ShadowKind::Dir, false, true, true) => &c.dir_instanced,
                    (ShadowKind::Dir, false, false, false) => &self.pipeline_shadow,
                    (ShadowKind::Dir, false, false, true) => &c.dir,
                    (ShadowKind::Spot, true, _, false) => &self.pipeline_shadow_spot_skinned,
                    (ShadowKind::Spot, true, _, true) => &c.spot_skinned,
                    (ShadowKind::Spot, false, true, false) => &self.pipeline_shadow_spot_instanced,
                    (ShadowKind::Spot, false, true, true) => &c.spot_instanced,
                    (ShadowKind::Spot, false, false, false) => &self.pipeline_shadow_spot,
                    (ShadowKind::Spot, false, false, true) => &c.spot,
                    (ShadowKind::Point, true, _, _) => &self.pipeline_shadow_point_skinned,
                    (ShadowKind::Point, false, true, _) => &self.pipeline_shadow_point_instanced,
                    (ShadowKind::Point, false, false, _) => &self.pipeline_shadow_point,
                };
                let cull = if p.kind == ShadowKind::Point {
                    shadow_cull_point(d.shadow_side)
                } else {
                    shadow_cull(d.shadow_side)
                };
                if skinned {
                    // The pose, not the bind pose: the skinned pipeline with
                    // the bones and the joint/weight buffer.
                    let (Some(bg), Some(attrs)) = (
                        skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()),
                        self.skin_attr_cache.get(&d.key),
                    ) else {
                        continue;
                    };
                    let base = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, base, d.map_variant);
                    spass.set_pipeline(&self.state_pipelines.cull_variant(&self.device, &base, cull));
                    spass.set_bind_group(1, bg, &[]);
                    spass.set_vertex_buffer(1, attrs.slice(..));
                } else {
                    let base = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, base, d.map_variant);
                    spass.set_pipeline(&self.state_pipelines.cull_variant(&self.device, &base, cull));
                    spass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    if instanced {
                        // Every instance, at its matrix and with its morph row.
                        spass.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        instances = 0..d.instance_count;
                    }
                }
                spass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                spass.set_bind_group(3, &self.shadow_env_bind_group, &[]);
                spass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    spass.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                } else {
                    spass.draw(draw_span(d.range, gm.vertex_count), instances);
                }
            }
        }

        // Screen-space refraction glass: when any Refract surface is present, the
        // opaque scene is rendered into the mipped ss-color capture (instead of
        // straight to the frame target); the glass pass below then refracts and
        // reflects that capture. Otherwise the main pass targets the frame view
        // directly, exactly as before.
        // A custom shader that asked for the opaque-scene capture needs the same
        // machinery the refraction glass does, so it turns it on too.
        // SUPPRESSED while rendering an environment cube. The glass path
        // copies the frame's depth into a screen-sized scratch texture, and a
        // cube face's depth is 48 pixels, not 7680 — wgpu refuses the copy
        // outright ("the entire texture must be copied when copying from depth
        // texture") and the whole render dies. A reflection probe has no use
        // for screen-space refraction anyway: it exists to light other
        // surfaces, and what it would refract is not in it.
        let wants_screen_space = draws.iter().any(|d| {
            d.topology == Topology::Refract
                || matches!(d.topology, Topology::Custom(_, _, true, ..))
        });
        // Screen-space refraction captures and marches the *whole* frame, so
        // it has no per-pane meaning: with a viewport or a partial clear those
        // draws are skipped (the main pass already leaves them out) rather
        // than drawn against the wrong pixels.
        if wants_screen_space
            && !vp_plan.legacy
            && !self.cube_pass
            && !self.viewport_state.warned_screen_space
        {
            self.viewport_state.warned_screen_space = true;
            log::warn!(
                "screen-space refraction / screen-space shader materials are not drawn \
                 with a viewport or scissor set"
            );
        }
        let refract_present = !self.cube_pass && vp_plan.legacy && wants_screen_space;
        let ss_format = if linear_framebuffer {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.color_format
        };
        if refract_present {
            let (w, h) = self.depth_size;
            self.ensure_ss_targets(w, h, ss_format);
        }

        // Compile any not-yet-seen custom-shader pipelines and build their
        // @group(1) user bind groups (buffers kept alive until submit). Hoisted
        // above the passes because a screen-space custom material draws in the
        // glass pass rather than the opaque one, and both need these.
        let mut user_bgs: Vec<Option<wgpu::BindGroup>> = Vec::new();
        user_bgs.resize_with(draws.len(), || None);
        let mut _custom_keep_alive: Vec<wgpu::Buffer> = Vec::new();
        let mut _custom_keep_alive_tex: Vec<wgpu::Texture> = Vec::new();
        for (i, d) in draws.iter().enumerate() {
            if let Topology::Custom(hash, transparent, screen_space, side) = d.topology {
                let sd = &shader_data[d.shader_idx];
                let key = (hash, linear_framebuffer, transparent, screen_space, side);
                let layout = if screen_space {
                    self.custom_ss_pipeline_layout.as_ref()
                } else {
                    self.custom_pipeline_layout.as_ref()
                };
                let Some(custom_layout) = layout else {
                    // No layout on this device — skip the object rather than
                    // submit a draw that cannot validate.
                    continue;
                };
                if self.custom_failed.contains(&key) {
                    continue;
                }
                if !self.custom_pipelines.contains_key(&key) {
                    let fmt = if linear_framebuffer {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        self.color_format
                    };
                    let pipeline = build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        fmt,
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        1,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    );
                    match pipeline {
                        Ok(p) => {
                            self.custom_pipelines.insert(key, p);
                        }
                        Err(e) => {
                            // One material is skipped; the rest of the frame
                            // draws. Previously this was a panic from inside
                            // wgpu, which took the program with it.
                            log::error!("shader material failed to compile:\n{e}");
                            self.custom_failed.insert(key);
                            continue;
                        }
                    }
                }
                // The encoded-blend pass's variant, into the `Unorm` view.
                let ekey = (hash, transparent, screen_space, side);
                if encoded_pass
                    && d.topology.encodes()
                    && !self.custom_encoded_pipelines.contains_key(&ekey)
                {
                    match build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        self.color_format.remove_srgb_suffix(),
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        1,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    ) {
                        Ok(p) => {
                            self.custom_encoded_pipelines.insert(ekey, p);
                        }
                        Err(e) => log::error!("shader material failed to compile:\n{e}"),
                    }
                }
                // The multisampled variant, for a frame drawn under MSAA: an
                // opaque material only (the blended passes draw single-sampled).
                let mkey = (hash, transparent, screen_space, side);
                if self.msaa > 1
                    && !transparent
                    && !screen_space
                    && !linear_framebuffer
                    && !self.custom_msaa_pipelines.contains_key(&mkey)
                {
                    match build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        self.color_format,
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        MSAA_SAMPLES,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    ) {
                        Ok(p) => {
                            self.custom_msaa_pipelines.insert(mkey, p);
                        }
                        Err(e) => log::error!("shader material failed to compile:\n{e}"),
                    }
                }
                let (bg, bufs, texs) = build_user_bind_group(
                    &self.device,
                    &self.queue,
                    &self.custom_mesh_bgl,
                    wgpu::BufferBinding {
                        buffer: self.mesh_arena.as_ref().expect("grown above"),
                        offset: self.per_mesh[i].0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<MeshUniforms>() as u64),
                    },
                    &sd.data,
                    &sd.storage,
                    &sd.textures,
                    &self.default_white_linear.view,
                    &self.default_data_tex_view,
                    &self.sampler_linear_repeat,
                    &sd.volumes,
                    &self.default_volume_view,
                    &self.sampler_volume,
                );
                _custom_keep_alive.extend(bufs);
                _custom_keep_alive_tex.extend(texs);
                user_bgs[i] = Some(bg);
            }
        }

        {
            let main_format = if linear_framebuffer {
                wgpu::TextureFormat::Rgba16Float
            } else {
                self.color_format
            };
            // `scene.background` is linear; the shared rule encodes it only
            // where the shader encodes too (a plain `Unorm` target), so a hex
            // background lands as that hex — see `super::clear_value`.
            let clear =
                super::clear_value(main_format, scene.background, scene.background_alpha);
            // Hardware MSAA on the direct-to-surface opaque pass. Falls back to
            // single-sample for render-to-texture, linear/postfx targets, and any
            // scene using glass/OIT/refraction/instanced/skinned/custom draws.
            // Not straight into the target when the colour is loaded (a partial
            // clear, or none): the resolve rewrites the whole target and would
            // erase the other panes. A partial clear goes through a pane-sized
            // target instead (`pane_msaa`); with no clear at all there is
            // nothing to start the pane target from, so that stays
            // single-sampled.
            let pane_key = pane_msaa
                .as_ref()
                .map(|p| (p.region.width, p.region.height, main_format, self.msaa));
            // OIT and refraction glass multisample on the whole target (not
            // in a pane), with a capture the colour can resolve into.
            let oit_msaa = self.pipeline_oit_msaa.is_some() && !linear_framebuffer;
            // Under MSAA the glass reads a single-sampled opaque depth: the
            // multisampled one resolved where a shader can read it, the
            // opaque scene's depth redrawn where it cannot (WebGL2).
            let refract_msaa = !linear_framebuffer && ss_format == self.color_format;
            let use_msaa = pane_msaa.is_some()
                || (self.msaa > 1
                    && vp_plan.clear == super::viewport::ClearMode::Full
                    && depth_src.is_none()
                    && (!linear_framebuffer || taa_src.is_some())
                    && (!refract_present || refract_msaa)
                    && draws.iter().all(|d| {
                        msaa_drawable(d.topology)
                            || (d.topology == Topology::Oit && oit_msaa)
                            || (d.topology == Topology::Refract && refract_msaa)
                    }));
            // Not over refraction glass: it is drawn single-sampled into the
            // frame, and resolving the multisampled colour again would erase it.
            self.overlay_msaa = use_msaa && pane_msaa.is_none() && !refract_present;
            self.last_frame_msaa = use_msaa;
            // three.js `polygonOffset`: the depth-biased pipelines this frame's
            // draws need, built once per (topology, target, offset).
            let bias_variant: u8 = if use_msaa {
                2
            } else if linear_framebuffer {
                1
            } else {
                0
            };
            // The encoded-blend pass's variant: the target's `Unorm` view,
            // single- or multisampled.
            let enc_variant: u8 = if use_msaa { 4 } else { 3 };
            if encoded_pass && use_msaa {
                self.ensure_msaa_targets(self.depth_size.0, self.depth_size.1);
            }
            for d in &draws {
                let in_encoded_pass = encoded_pass && d.topology == Topology::TriangleAlpha;
                let (factor, units) = match d.polygon_offset {
                    Some(o) => o,
                    None if in_encoded_pass => (0f32.to_bits(), 0),
                    None => continue,
                };
                let Some(code) = biased_topology(d.topology) else {
                    continue;
                };
                let variant = if in_encoded_pass { enc_variant } else { bias_variant };
                let key = (code, variant, factor, units);
                if !self.biased_pipelines.contains_key(&key) {
                    let (format, samples) = match variant {
                        4 => (self.color_format.remove_srgb_suffix(), MSAA_SAMPLES),
                        3 => (self.color_format.remove_srgb_suffix(), 1),
                        2 => (self.color_format, MSAA_SAMPLES),
                        1 => (wgpu::TextureFormat::Rgba16Float, 1),
                        _ => (self.color_format, 1),
                    };
                    let p = build_biased_pipeline(
                        &self.device,
                        &self.mesh_shader,
                        &self.mesh_pipeline_layout,
                        format,
                        samples,
                        code,
                        f32::from_bits(factor),
                        units,
                        &mut self.state_pipelines,
                    );
                    self.biased_pipelines.insert(key, p);
                }
            }
            if let Some(key) = pane_key {
                self.viewport_state
                    .ensure_pane_target(&self.device, key, DEPTH_FORMAT);
            } else if use_msaa {
                self.ensure_msaa_targets(self.depth_size.0, self.depth_size.1);
            }
            // The pane's own plan when drawing into a pane target.
            let main_plan = pane_msaa.as_ref().map(|p| p.local()).unwrap_or(vp_plan);
            if let super::viewport::ClearMode::Rect(_) = vp_plan.clear {
                self.viewport_state
                    .ensure_clear_pipeline(&self.device, main_format, Some(DEPTH_FORMAT));
            }
            // Screen-space glass redirects the opaque pass into the ss-color capture
            // (mip 0); MSAA renders into the multisampled color and resolves into the
            // frame view; otherwise it targets the frame view directly.
            let (main_color_view, msaa_resolve, depth_pass_view): (
                &wgpu::TextureView,
                Option<&wgpu::TextureView>,
                &wgpu::TextureView,
            ) = if let Some(pane) = pane_key
                .as_ref()
                .and_then(|k| self.viewport_state.pane_target(k))
            {
                (&pane.msaa_color, Some(&pane.resolve), &pane.msaa_depth)
            } else if use_msaa {
                let (_, _, cv, dv) = self.msaa_targets.as_ref().unwrap();
                // Under refraction glass the opaque scene resolves into the
                // capture the glass refracts, as it is drawn there unsampled.
                let resolve = if refract_present {
                    &self.ss_color.as_ref().unwrap().mip_render_views[0]
                } else {
                    target_view
                };
                (cv, Some(resolve), dv)
            } else if refract_present {
                (
                    &self.ss_color.as_ref().unwrap().mip_render_views[0],
                    None,
                    &self.depth_view,
                )
            } else {
                (target_view, None, &self.depth_view)
            };

            // Timestamps go on the main pass alone. It is where the scene's
            // geometry is actually shaded — shadows, post-fx and the glass
            // passes are separate costs and averaging them together would hide
            // exactly the change anyone is trying to measure.
            let main_timestamps = self.timer.as_ref().map(|t| t.writes());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: main_color_view,
                    depth_slice: None,
                    resolve_target: msaa_resolve,
                    ops: wgpu::Operations {
                        load: if main_plan.clear == super::viewport::ClearMode::Full {
                            wgpu::LoadOp::Clear(clear)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_pass_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth_clear()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: main_timestamps,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // The partial clear first: the clear colour over the scissor
            // rectangle, as `gl.clear` does with the scissor test on.
            if let super::viewport::ClearMode::Rect(r) = main_plan.clear {
                self.viewport_state.fill_rect(&mut pass, main_format, true, r, clear);
            }
            main_plan.apply(&mut pass);
            // Alpha-to-coverage pipelines store the blend constant as their
            // alpha: opaque. (The partial clear above sets its own.)
            pass.set_blend_constant(wgpu::Color::WHITE);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(3, &self.env_bind_group, &[]);
            let mut last_topology: Option<Topology> = None;
            let mut last_bias: Option<(u32, i32)> = None;
            let mut last_state = crate::materials::RenderState::DEFAULT;
            let mut last_variant: Option<super::material_maps::MapVariant> = None;
            let mut last_side: u32 = u32::MAX;
            // Group 3 is the scene's environment unless a material has its own.
            let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
            for (i, d) in draws.iter().enumerate() {
                // A custom draw on a trimmed-environment backend (WebGL2) binds
                // the custom pipeline's own group 3; the full environment its
                // standard counterpart binds does not match that pipeline layout.
                let custom_env = self.custom_env_bind_group.is_some()
                    && matches!(d.topology, Topology::Custom(..));
                if (d.env_group, custom_env) != env_bound {
                    pass.set_bind_group(
                        3,
                        if custom_env {
                            self.custom_env_bind_group.as_ref().unwrap()
                        } else {
                            d.env_group
                                .and_then(|k| self.material_env_groups.get(&k))
                                .map_or(&self.env_bind_group, |(_, g)| g)
                        },
                        &[],
                    );
                    env_bound = (d.env_group, custom_env);
                }
                // Off-screen. The shadow passes above deliberately did draw it:
                // a caster the camera cannot see still casts into the view.
                if !d.in_frustum {
                    continue;
                }
                // Writes neither colour nor depth: nothing to see here (a
                // shadow-only caster — fur's and hair's). Skipped rather than
                // drawn for nothing; and on WebGL2 a colour-masked draw left
                // the whole frame black.
                if !d.state.color_write && d.state.depth_write == Some(false) {
                    continue;
                }
                // OIT + screen-space-refraction surfaces are drawn in dedicated
                // later passes, not here.
                if d.topology == Topology::Oit
                    || d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, ..))
                {
                    continue;
                }
                // Objects opted out of glass capture draw *over* the glass (a
                // later overlay pass), so they're not refracted — skip here.
                if refract_present && !d.refract_capture {
                    continue;
                }
                // Blended draws on an sRGB target: the encoded-blend pass.
                if encoded_pass && d.topology.encodes() {
                    continue;
                }
                if last_topology != Some(d.topology)
                    || last_bias != d.polygon_offset
                    || last_state != d.state
                    || last_variant != d.map_variant
                    || last_side != d.side
                {
                    let biased = d.polygon_offset.and_then(|(factor, units)| {
                        let code = biased_topology(d.topology)?;
                        self.biased_pipelines.get(&(code, bias_variant, factor, units))
                    });
                    let pipe = if let Some(p) = biased {
                        p
                    } else if use_msaa {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri_msaa,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha_msaa,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull_msaa,
                            Topology::Sky => &self.pipeline_tri_sky_msaa,
                            Topology::TriangleWire => &self.pipeline_tri_wire_msaa,
                            Topology::Line => &self.pipeline_line_msaa,
                            Topology::Point => &self.pipeline_point_msaa,
                            Topology::WideLine => &self.pipeline_wide_line_msaa,
                            Topology::GlassDepth => &self.pipeline_glass_depth_msaa,
                            Topology::GlassColor => &self.pipeline_glass_color_msaa,
                            Topology::GlassTint => &self.pipeline_glass_tint_msaa,
                            Topology::Skinned => &self.pipeline_skinned_msaa,
                            Topology::Instanced => &self.pipeline_instanced_msaa,
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_msaa_pipelines.get(&(h, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                            // gated by `use_msaa`; other topologies never occur here
                            _ => &self.pipeline_tri_msaa,
                        }
                    } else if linear_framebuffer {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri_f16,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha_f16,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull_f16,
                            Topology::GlassDepth => &self.pipeline_glass_depth_f16,
                            Topology::GlassColor => &self.pipeline_glass_color_f16,
                            Topology::GlassTint => &self.pipeline_glass_tint_f16,
                            Topology::Sky => &self.pipeline_tri_sky_f16,
                            Topology::TriangleWire => &self.pipeline_tri_wire_f16,
                            Topology::Line => &self.pipeline_line_f16,
                            Topology::Point => &self.pipeline_point_f16,
                            Topology::PointAlpha => &self.pipeline_point_alpha_f16,
                            Topology::WideLine => &self.pipeline_wide_line_f16,
                            Topology::WideLineAlpha => &self.pipeline_wide_line_alpha_f16,
                            Topology::Sprite => &self.pipeline_sprite_f16,
                            Topology::Instanced => &self.pipeline_instanced_f16,
                            Topology::InstancedWire => &self.pipeline_instanced_wire_f16,
                            Topology::InstancedAlpha => &self.pipeline_instanced_alpha_f16,
                            Topology::Skinned => &self.pipeline_skinned_f16,
                            Topology::SkinnedWire => &self.pipeline_skinned_wire_f16,
                            Topology::SkinnedAlpha => &self.pipeline_skinned_alpha_f16,
                            Topology::Oit | Topology::Refract => continue,
                            // A shader that would not compile has no
                            // pipeline. Skipping the draw costs one object;
                            // indexing the map costs the process.
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_pipelines.get(&(h, true, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                        }
                    } else {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull,
                            Topology::GlassDepth => &self.pipeline_glass_depth,
                            Topology::GlassColor => &self.pipeline_glass_color,
                            Topology::GlassTint => &self.pipeline_glass_tint,
                            Topology::Sky => &self.pipeline_tri_sky,
                            Topology::TriangleWire => &self.pipeline_tri_wire,
                            Topology::Line => &self.pipeline_line,
                            Topology::Point => &self.pipeline_point,
                            Topology::PointAlpha => &self.pipeline_point_alpha,
                            Topology::WideLine => &self.pipeline_wide_line,
                            Topology::WideLineAlpha => &self.pipeline_wide_line_alpha,
                            Topology::Sprite => &self.pipeline_sprite,
                            Topology::Instanced => &self.pipeline_instanced,
                            Topology::InstancedWire => &self.pipeline_instanced_wire,
                            Topology::InstancedAlpha => &self.pipeline_instanced_alpha,
                            Topology::Skinned => &self.pipeline_skinned,
                            Topology::SkinnedWire => &self.pipeline_skinned_wire,
                            Topology::SkinnedAlpha => &self.pipeline_skinned_alpha,
                            Topology::Oit | Topology::Refract => continue,
                            // A shader that would not compile has no
                            // pipeline. Skipping the draw costs one object;
                            // indexing the map costs the process.
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_pipelines.get(&(h, false, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                        }
                    };
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        pipe,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                    let pipe = match side_cull(d.topology.is_faces(), d.side) {
                        Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                        None => pipe,
                    };
                    pass.set_pipeline(&pipe);
                    last_side = d.side;
                    last_topology = Some(d.topology);
                    last_bias = d.polygon_offset;
                    last_state = d.state;
                    last_variant = d.map_variant;
                }
                let gm = if d.topology == Topology::Sprite {
                    &self.sprite_quad
                } else {
                    &self.geom_cache[&d.key].mesh
                };
                if d.topology.is_skinned() {
                    // Skinned pipeline uses the combined mesh+skin BGL at group(1).
                    if let Some(bg) = skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()) {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    if let Some(skin_attrs) = self.skin_attr_cache.get(&d.key) {
                        pass.set_vertex_buffer(1, skin_attrs.slice(..));
                    }
                } else if let Topology::Custom(..) = d.topology {
                    // Custom-shader draws use the combined mesh+user bind group
                    // at @group(1) (mesh uniform + user data/storage), and a
                    // group 2 holding only what the shared vertex entry needs —
                    // a `ShaderMaterial` has no standard maps, and declaring the
                    // eight it cannot use is what put this layout over WebGPU's
                    // per-stage texture limit.
                    if let Some(bg) = &user_bgs[i] {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(
                        2,
                        custom_morph_bgs[i].as_ref().unwrap_or(&self.custom_tex_bind_group),
                        &[],
                    );
                } else {
                    pass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    pass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                }
                // Wide lines: a capsule (six corners) per segment in the range —
                // an index span, or a vertex span, two to a segment — the segment's
                // start through slot 0 and its end a vertex further through slot 1.
                if d.topology.is_wide_lines() {
                    let Some(buf) = &gm.segment_buffer else { continue };
                    let span = draw_span(d.range, gm.segment_vertices);
                    let segs = span.start / 2..span.end / 2;
                    if segs.is_empty() {
                        continue;
                    }
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                    pass.draw(0..6, segs);
                    continue;
                }
                pass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                // Points: a quad's six corners for each point in the range.
                if d.topology.is_points() {
                    pass.draw(0..6, draw_span(d.range, gm.vertex_count));
                    continue;
                }
                // A `ShaderMaterial` on an `InstancedMesh` reads the instance
                // buffer too, and its custom columns from slot 2.
                let custom_instanced = matches!(d.topology, Topology::Custom(..))
                    && shader_data[d.shader_idx].instanced;
                let instance_count = if d.topology.is_instanced() || custom_instanced {
                    pass.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                    if custom_instanced {
                        let custom = &instance_buffers[shader_data[d.shader_idx].custom_buf];
                        pass.set_vertex_buffer(2, custom.slice(..));
                    }
                    d.instance_count
                } else {
                    1
                };
                // Wire topologies draw the line-list index buffer: polygon
                // outlines, or three.js's a,b,b,c,c,a per triangle without a
                // polygon table (`BufferGeometry::wireframe_indices`).
                // A partial span (a group, a draw range) has a line list of
                // its own (`BufferGeometry::wireframe_indices_in`).
                let (ib_opt, idx_range) = if d.topology.is_wire() {
                    match d.range {
                        Some((a, b)) => {
                            let version = self.geom_cache.get(&d.key).map(|e| e.version);
                            match version.and_then(|v| self.wire_span_cache.get(&(d.key, v, a, b))) {
                                Some((buf, n)) => (Some(buf), 0..*n),
                                None => (None, 0..0),
                            }
                        }
                        None => (gm.wire_index_buffer.as_ref(), 0..gm.wire_index_count),
                    }
                } else {
                    (gm.index_buffer.as_ref(), draw_span(d.range, gm.index_count))
                };
                if let Some(ib) = ib_opt {
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(idx_range, 0, 0..instance_count);
                } else if d.topology.is_wire() && d.range.is_some() {
                    // Nothing to outline in this span.
                } else {
                    pass.draw(draw_span(d.range, gm.vertex_count), 0..instance_count);
                }
            }
            drop(pass);

            // The encoded-blend pass (see `encoded_pass`): the blended meshes,
            // back to front, into a `Unorm` view of the target (or of the
            // multisampled colour) with the shader encoding, so the hardware
            // blend mixes sRGB-encoded values as three.js's canvas does.
            if let (true, Some(tex)) = (encoded_pass, encoded_target) {
                let unorm = tex.format().remove_srgb_suffix();
                // (Not on the scratch path: there the device cannot take it.)
                let target_unorm = (!encoded_scratch).then(|| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(unorm),
                        ..Default::default()
                    })
                });
                // Where the main pass drew: a pane's own multisampled target,
                // the shared multisampled colour, the refraction capture, or
                // the target itself.
                let pane = pane_key
                    .as_ref()
                    .and_then(|k| self.viewport_state.pane_target(k));
                let color_view = if encoded_scratch {
                    self.encoded_scratch.as_ref().map(|(_, _, v)| v)
                } else if let Some(pane) = pane {
                    pane.msaa_color_unorm.as_ref()
                } else if use_msaa {
                    self.msaa_color_unorm.as_ref()
                } else if refract_present {
                    self.ss_color.as_ref().and_then(|c| c.mip0_unorm.as_ref())
                } else {
                    target_unorm.as_ref()
                };
                let target_srgb = encoded_scratch.then(|| tex.create_view(&Default::default()));
                if let (Some(color_view), Some(enc)) = (color_view, self.encoded_blend_pipelines.as_ref()) {
                    if let Some(t) = &target_srgb {
                        self.encoded_copy_pass(&mut encoder, false, t, color_view);
                    }
                    let mut ep = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers encoded blend pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_pass_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    main_plan.apply(&mut ep);
                    ep.set_bind_group(0, &self.frame_bind_group, &[]);
                    ep.set_bind_group(3, &self.env_bind_group, &[]);
                    // Group 3 is the scene's environment unless a material has its own.
                    let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                    for (i, d) in draws.iter().enumerate() {
                        // A custom draw on a trimmed-environment backend (WebGL2) binds
                        // the custom pipeline's own group 3; the full environment its
                        // standard counterpart binds does not match that pipeline layout.
                        let custom_env = self.custom_env_bind_group.is_some()
                            && matches!(d.topology, Topology::Custom(..));
                        if (d.env_group, custom_env) != env_bound {
                            ep.set_bind_group(
                                3,
                                if custom_env {
                                    self.custom_env_bind_group.as_ref().unwrap()
                                } else {
                                    d.env_group
                                        .and_then(|k| self.material_env_groups.get(&k))
                                        .map_or(&self.env_bind_group, |(_, g)| g)
                                },
                                &[],
                            );
                            env_bound = (d.env_group, custom_env);
                        }
                        if !d.in_frustum || !d.topology.encodes() {
                            continue;
                        }
                        // Opted out of the refraction capture: the overlay
                        // pass draws it, over the glass.
                        if refract_present && !d.refract_capture {
                            continue;
                        }
                        let pipe = match d.topology {
                            Topology::TriangleAlpha => {
                                let (factor, units) = d.polygon_offset.unwrap_or((0f32.to_bits(), 0));
                                self.biased_pipelines.get(&(1, enc_variant, factor, units))
                            }
                            Topology::Sprite => Some(&enc[0]),
                            Topology::PointAlpha => Some(&enc[3]),
                            Topology::WideLineAlpha => Some(&enc[7]),
                            Topology::InstancedAlpha => Some(&enc[1]),
                            Topology::SkinnedAlpha => Some(&enc[2]),
                            Topology::GlassDepth => Some(match (&self.encoded_glass_msaa, use_msaa) {
                                (Some(m), true) => &m[0],
                                _ => &enc[4],
                            }),
                            Topology::GlassColor => Some(match (&self.encoded_glass_msaa, use_msaa) {
                                (Some(m), true) => &m[1],
                                _ => &enc[5],
                            }),
                            Topology::GlassTint => Some(match (&self.encoded_glass_msaa, use_msaa) {
                                (Some(m), true) => &m[2],
                                _ => &enc[6],
                            }),
                            Topology::Custom(h, tr, ss, sd) => {
                                self.custom_encoded_pipelines.get(&(h, tr, ss, sd))
                            }
                            _ => None,
                        };
                        let Some(pipe) = pipe else { continue };
                        let pipe = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, pipe, d.map_variant);
                        let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                        let pipe = match side_cull(d.topology.is_faces(), d.side) {
                            Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                            None => pipe,
                        };
                        ep.set_pipeline(&pipe);
                        let gm = if d.topology == Topology::Sprite {
                            &self.sprite_quad
                        } else {
                            &self.geom_cache[&d.key].mesh
                        };
                        if d.topology.is_skinned() {
                            let Some(bg) = skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()) else {
                                continue;
                            };
                            let Some(skin_attrs) = self.skin_attr_cache.get(&d.key) else {
                                continue;
                            };
                            ep.set_bind_group(1, bg, &[]);
                            ep.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                            ep.set_vertex_buffer(1, skin_attrs.slice(..));
                        } else if let Topology::Custom(..) = d.topology {
                            let Some(bg) = &user_bgs[i] else { continue };
                            ep.set_bind_group(1, bg, &[]);
                            ep.set_bind_group(
                                2,
                                custom_morph_bgs[i].as_ref().unwrap_or(&self.custom_tex_bind_group),
                                &[],
                            );
                        } else {
                            ep.set_bind_group(1, &self.per_mesh[i].1, &[]);
                            ep.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        }
                        // Wide lines: a capsule (six corners) per segment in the range —
                        // an index span, or a vertex span, two to a segment — the segment's
                        // start through slot 0 and its end a vertex further through slot 1.
                        if d.topology.is_wide_lines() {
                            let Some(buf) = &gm.segment_buffer else { continue };
                            let span = draw_span(d.range, gm.segment_vertices);
                            let segs = span.start / 2..span.end / 2;
                            if segs.is_empty() {
                                continue;
                            }
                            ep.set_vertex_buffer(0, buf.slice(..));
                            ep.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                            ep.draw(0..6, segs);
                            continue;
                        }
                        ep.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if d.topology.is_points() {
                            ep.draw(0..6, draw_span(d.range, gm.vertex_count));
                            continue;
                        }
                        let custom_instanced = matches!(d.topology, Topology::Custom(..))
                            && shader_data[d.shader_idx].instanced;
                        let instances = if d.topology.is_instanced() || custom_instanced {
                            ep.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                            if custom_instanced {
                                let custom = &instance_buffers[shader_data[d.shader_idx].custom_buf];
                                ep.set_vertex_buffer(2, custom.slice(..));
                            }
                            d.instance_count
                        } else {
                            1
                        };
                        if let Some(ib) = &gm.index_buffer {
                            ep.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            ep.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..instances);
                        } else {
                            ep.draw(draw_span(d.range, gm.vertex_count), 0..instances);
                        }
                    }
                    drop(ep);
                    if let Some(t) = &target_srgb {
                        self.encoded_copy_pass(&mut encoder, true, color_view, t);
                    }
                    // Multisampled: resolve again, in linear light, now that
                    // the samples hold the blended surfaces too.
                    if use_msaa {
                        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("threers encoded blend resolve"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: main_color_view,
                                depth_slice: None,
                                resolve_target: msaa_resolve,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                    }
                }
            }
        }

        // A pane drawn into its own multisampled target: copy its resolved
        // picture into the clear rectangle, and only there.
        if let Some(p) = &pane_msaa {
            let key = (p.region.width, p.region.height, self.color_format, self.msaa);
            self.viewport_state.composite_pane(
                &mut encoder,
                target_view,
                self.color_format,
                &key,
                p.region,
            );
        }

        // Which OIT draws composite where. Without refraction, all of them
        // over the frame. With it, each where the refraction glass would see
        // it: into the capture just before the first group of glass that
        // lies wholly in front of it (so that glass refracts it), or, nearer
        // than every group, over the finished glass.
        let (refract_groups, refract_group) = {
            let mut order: Vec<usize> = (0..draws.len())
                .filter(|&i| refract_present && draws[i].in_frustum && draws[i].topology == Topology::Refract)
                .collect();
            order.sort_by(|&a, &b| draws[a].view_z.partial_cmp(&draws[b].view_z).unwrap_or(std::cmp::Ordering::Equal));
            let groups = self.refraction_layers.clamp(1, order.len().max(1));
            let mut group = vec![usize::MAX; draws.len()];
            for (k, &i) in order.iter().enumerate() {
                group[i] = k * groups / order.len();
            }
            (groups, group)
        };
        let oit_phase: Vec<usize> = {
            // Each group's farthest object, as a view depth (more negative is
            // farther).
            let mut farthest = vec![f32::INFINITY; refract_groups];
            for (i, &g) in refract_group.iter().enumerate() {
                if g != usize::MAX {
                    farthest[g] = farthest[g].min(draws[i].view_z);
                }
            }
            draws
                .iter()
                .map(|d| {
                    if d.topology != Topology::Oit {
                        usize::MAX
                    } else if !refract_present {
                        0
                    } else {
                        (0..refract_groups).find(|&g| d.view_z <= farthest[g]).unwrap_or(refract_groups)
                    }
                })
                .collect()
        };
        // One OIT phase: accumulate its draws against the depth so far, then
        // composite them into `$target` — the frame, or the refraction
        // capture — multisampled when `$allow_msaa` and the frame is.
        macro_rules! oit_phase {
            ($phase:expr, $target:expr, $allow_msaa:expr) => {{
        // --- Weighted-blended OIT: accumulate the OIT surfaces, then resolve
        // (composite) over the opaque image already in `target_view`. ---
        // Not during a probe, and this one bites hard. The order-independent
        // transparency targets are sized from `depth_size`, which a cube pass
        // has set to the face's 48 pixels — so a probe RESIZES them, and the
        // next real frame resolves a 48-pixel transparency buffer stretched
        // across the whole picture. It reads as the entire vehicle going
        // translucent, which looks like an alpha bug in the materials and is
        // nothing of the kind.
        if !self.cube_pass
            && self.pipeline_oit.is_some()
            && draws.iter().enumerate().any(|(i, d)| d.topology == Topology::Oit && oit_phase[i] == $phase)
        {
            let (tw, th) = self.depth_size;
            self.ensure_oit_targets(tw, th);
            // Multisampled when the opaque pass was: against its depth, into
            // samples resolved at the end of the pass.
            let msaa = $allow_msaa && self.last_frame_msaa && pane_msaa.is_none() && self.pipeline_oit_msaa.is_some();
            if msaa {
                self.ensure_oit_msaa_targets(tw, th);
            }
            let msaa_views = msaa
                .then(|| Some((self.oit_msaa_targets.as_ref()?, self.msaa_targets.as_ref()?)))
                .flatten();
            if let Some((_, _, _, accum_view, _, reveal_view, resolve_bg)) =
                self.oit_targets.as_ref()
            {
                // 1) Accumulation pass (accum cleared to 0, revealage cleared to 1).
                {
                    let mut opass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit accum pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: msaa_views.map_or(accum_view, |((_, _, a, _), _)| a),
                                depth_slice: None,
                                resolve_target: msaa_views.map(|_| accum_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: msaa_views.map_or(reveal_view, |((_, _, _, r), _)| r),
                                depth_slice: None,
                                resolve_target: msaa_views.map(|_| reveal_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 1.0,
                                        g: 1.0,
                                        b: 1.0,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: msaa_views.map_or(&self.depth_view, |(_, (_, _, _, dv))| dv),
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let pipe = if msaa_views.is_some() { &self.pipeline_oit_msaa } else { &self.pipeline_oit };
                    let Some(oit_pipe) = pipe.as_ref() else {
                        unreachable!("oit pass entered without a pipeline")
                    };
                    vp_plan.apply(&mut opass);
                    opass.set_pipeline(oit_pipe);
                    opass.set_bind_group(0, &self.frame_bind_group, &[]);
                    opass.set_bind_group(3, &self.env_bind_group, &[]);
                    // Group 3 is the scene's environment unless a material has its own.
                    let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                    for (i, d) in draws.iter().enumerate() {
                        // A custom draw on a trimmed-environment backend (WebGL2) binds
                        // the custom pipeline's own group 3; the full environment its
                        // standard counterpart binds does not match that pipeline layout.
                        let custom_env = self.custom_env_bind_group.is_some()
                            && matches!(d.topology, Topology::Custom(..));
                        if (d.env_group, custom_env) != env_bound {
                            opass.set_bind_group(
                                3,
                                if custom_env {
                                    self.custom_env_bind_group.as_ref().unwrap()
                                } else {
                                    d.env_group
                                        .and_then(|k| self.material_env_groups.get(&k))
                                        .map_or(&self.env_bind_group, |(_, g)| g)
                                },
                                &[],
                            );
                            env_bound = (d.env_group, custom_env);
                        }
                        if !d.in_frustum {
                            continue;
                        }
                        if d.topology != Topology::Oit || oit_phase[i] != $phase {
                            continue;
                        }
                        // The draw's group 2 is its material variant's, so
                        // the pipeline has to be too.
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            oit_pipe,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(
                            &self.device,
                            &pipe,
                            &crate::materials::RenderState::DEFAULT,
                        );
                        opass.set_pipeline(&pipe);
                        let gm = &self.geom_cache[&d.key].mesh;
                        opass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                        opass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        opass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if let Some(ib) = &gm.index_buffer {
                            opass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            opass.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                        } else {
                            opass.draw(draw_span(d.range, gm.vertex_count), 0..1);
                        }
                    }
                }
                // 2) Resolve over the opaque image: what is behind times the
                //    layers' transmittance, then plus their light.
                {
                    let resolve_pipe = if msaa_views.is_some() {
                        &self.pipeline_oit_resolve_msaa
                    } else if linear_framebuffer {
                        &self.pipeline_oit_resolve_f16
                    } else {
                        &self.pipeline_oit_resolve
                    };
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit resolve pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: msaa_views.map_or($target, |(_, (_, _, cv, _))| cv),
                            depth_slice: None,
                            resolve_target: msaa_views.map(|_| $target),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if !vp_plan.legacy {
                        // Full-screen composite: confine it to this pane.
                        let sc = vp_plan.scissor;
                        rpass.set_scissor_rect(sc.x, sc.y, sc.width, sc.height);
                    }
                    rpass.set_bind_group(0, resolve_bg, &[]);
                    for pipe in resolve_pipe {
                        rpass.set_pipeline(pipe);
                        rpass.draw(0..3, 0..1);
                    }
                }
            }
        }

            }};
        }
        if !refract_present {
            oit_phase!(0, target_view, true);
        }

        // A multisampled opaque pass left its depth multisampled: resolve it
        // into the single-sampled one — the renderer's own, which refraction
        // glass tests against and its march copies, or a multisampled render
        // target's, which post-processing reads (`RenderTarget::samples`).
        let depth_out = self.last_frame_msaa && pane_msaa.is_none() && (refract_present || self.msaa_depth_out.is_some());
        if depth_out && (self.depth_resolve.is_none() || self.force_depth_replay) {
            // The camera's view-projection as the replay's pass uniform.
            if self.depth_replay_pass.is_none() {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers depth replay uniform"),
                    size: std::mem::size_of::<ShadowPassGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers depth replay bg"),
                    layout: &self.shadow_pass_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.frame_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: buffer.as_entire_binding() },
                    ],
                });
                self.depth_replay_pass = Some((buffer, bg));
            }
            let (replay_buffer, replay_bg) = self.depth_replay_pass.as_ref().unwrap();
            let replay_u = ShadowPassGpu { vp: frame_u.view_proj, light: [0.0; 4] };
            self.queue.write_buffer(replay_buffer, 0, bytemuck::bytes_of(&replay_u));
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers depth replay pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth_clear()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, replay_bg, &[0]);
            rp.set_bind_group(3, &self.shadow_env_bind_group, &[]);
            for (i, d) in draws.iter().enumerate() {
                if !d.in_frustum {
                    continue;
                }
                // What the opaque pass wrote depth for. Lines, points,
                // sprites and custom shaders are left out: they rarely
                // stand behind glass, and have no depth-only pipeline.
                let skinned = d.topology == Topology::Skinned;
                let instanced = d.topology == Topology::Instanced;
                if !(matches!(d.topology, Topology::Triangle | Topology::TriangleNoCull) || skinned || instanced) {
                    continue;
                }
                let cut = d.alpha_test > 0.0 && (d.slots.map.is_some() || d.slots.alpha_map.is_some());
                let base = &self.depth_replay[usize::from(cut) * 3 + if skinned { 1 } else if instanced { 2 } else { 0 }];
                let gm = &self.geom_cache[&d.key].mesh;
                let mut instances = 0..1;
                if skinned {
                    let (Some(bg), Some(attrs)) = (
                        skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()),
                        self.skin_attr_cache.get(&d.key),
                    ) else {
                        continue;
                    };
                    rp.set_bind_group(1, bg, &[]);
                    rp.set_vertex_buffer(1, attrs.slice(..));
                } else {
                    rp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    if instanced {
                        rp.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        instances = 0..d.instance_count;
                    }
                }
                let pipe = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, base, d.map_variant);
                // The material's depth state, and with reversed depth the
                // flipped compare, as the opaque pass drew it.
                let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                let pipe = match side_cull(true, d.side) {
                    Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                    None => pipe,
                };
                rp.set_pipeline(&pipe);
                rp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                rp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                } else {
                    rp.draw(draw_span(d.range, gm.vertex_count), instances);
                }
            }
        } else if depth_out {
            if let (Some((bgl, pipe)), Some((_, _, _, msaa_depth))) = (&self.depth_resolve, &self.msaa_targets) {
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers depth resolve bg"),
                    layout: bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(msaa_depth),
                    }],
                });
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers depth resolve pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.depth_clear()),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipe);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..3, 0..1);
            }
        }
        // --- Screen-space refraction glass: build the ss-color mip chain from
        // the opaque capture, blit the sharp capture to the frame target, then
        // draw the glass surface sampling it for refraction + reflection. ---
        //
        if refract_present {
            // The refracting objects back to front, in `refract_groups` groups
            // (worked out above, with the OIT phases).
            if refract_groups > 1 {
                self.ensure_ss_layer();
            }
            // Overlapping glass, in layers (`set_refraction_layers`): the
            // refracting objects back to front in up to that many groups,
            // each refracting a capture of everything drawn before it — the
            // opaque scene for the farthest, that and the farther glass for
            // the next. Every group but the last draws into a scratch target
            // copied back into the capture; the last draws into the frame.
            for group in 0..refract_groups {
            let last_group = group + 1 == refract_groups;
            if group > 0 {
                // What the previous group drew is the next one's capture.
                let (ssc, (_, _, layer_tex, _)) = (self.ss_color.as_ref().unwrap(), self.ss_layer.as_ref().unwrap());
                encoder.copy_texture_to_texture(
                    layer_tex.as_image_copy(),
                    ssc.texture.as_image_copy(),
                    wgpu::Extent3d { width: ssc.width, height: ssc.height, depth_or_array_layers: 1 },
                );
            }
            // OIT surfaces behind this group's glass, into its capture.
            oit_phase!(group, &self.ss_color.as_ref().unwrap().mip_render_views[0], false);
            let layer_target: &wgpu::TextureView = if last_group {
                target_view
            } else {
                &self.ss_layer.as_ref().unwrap().3
            };
            let ssc = self.ss_color.as_ref().unwrap();
            // Copy the opaque depth into the sampleable ss_depth so the glass
            // shader can ray-march it for refraction/reflection.
            if let (Some((_, _, _, ss_dview)), Some((bgl, pipe, sampler))) = (&self.ss_depth, &self.depth_copy) {
                // where depth cannot be copied: drawn across instead
                let src = depth_src.or(self.msaa_depth_out.as_ref()).unwrap_or(&self.depth_texture);
                let src_view = src.create_view(&wgpu::TextureViewDescriptor { aspect: wgpu::TextureAspect::DepthOnly, ..Default::default() });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers depth copy bg"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                    ],
                });
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers depth copy pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: ss_dview,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(self.depth_clear()), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipe);
                rp.set_bind_group(0, &bg, &[]);
                rp.draw(0..3, 0..1);
            } else if let Some((_, _, ss_dtex, _)) = &self.ss_depth {
                let src = depth_src.or(self.msaa_depth_out.as_ref()).unwrap_or(&self.depth_texture);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: ss_dtex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::DepthOnly,
                    },
                    wgpu::Extent3d {
                        width: ssc.width,
                        height: ssc.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let ss_depth_view = &self.ss_depth.as_ref().unwrap().3;
            let ss_back_depth_view = &self.ss_back_depth.as_ref().unwrap().3;
            // Two-surface: render glass back faces (cull front) into ss_back_depth
            // so the shader can derive per-pixel glass thickness.
            {
                let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers ss back-depth pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: ss_back_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.depth_clear()),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                bp.set_pipeline(&self.pipeline_ss_back_depth);
                bp.set_bind_group(0, &self.frame_bind_group, &[]);
                for (i, d) in draws.iter().enumerate() {
                    if !d.in_frustum {
                        continue;
                    }
                    if d.topology != Topology::Refract || refract_group[i] != group {
                        continue;
                    }
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        &self.pipeline_ss_back_depth,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(
                        &self.device,
                        &pipe,
                        &crate::materials::RenderState::DEFAULT,
                    );
                    bp.set_pipeline(&pipe);
                    let gm = &self.geom_cache[&d.key].mesh;
                    bp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    bp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    bp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if let Some(ib) = &gm.index_buffer {
                        bp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        bp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                    } else {
                        bp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                    }
                }
            }
            let blit_pipe = if linear_framebuffer {
                &self.pipeline_ss_blit_f16
            } else {
                &self.pipeline_ss_blit
            };
            // 1) Downsample chain: build mip m by box-filtering mip m-1.
            for m in 1..ssc.mips as usize {
                let src_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers ss mip src"),
                    layout: &self.ss_blit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &ssc.mip_sample_views[m - 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                        },
                    ],
                });
                let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers ss mip gen"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ssc.mip_render_views[m],
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
                mp.set_pipeline(blit_pipe);
                mp.set_bind_group(0, &src_bg, &[]);
                mp.draw(0..3, 0..1);
            }
            // 2) Blit the sharp capture (mip 0) to the frame target so non-glass
            //    pixels show through; the glass draws over it next.
            {
                let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers ss blit src"),
                    layout: &self.ss_blit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&ssc.mip_sample_views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                        },
                    ],
                });
                let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers ss blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: layer_target,
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
                bp.set_pipeline(blit_pipe);
                bp.set_bind_group(0, &blit_bg, &[]);
                bp.draw(0..3, 0..1);
            }
            // 3) group(3) bind group with the real ss-color at binding 9 (env cube
            //    + shadows mirror the frame's env bind group).
            // The scene environment as the frame's group reads it: prefiltered
            // (earlier this frame).
            let scene_env_cube = scene.environment.as_ref().map(|e| self.prefiltered_cached(e));
            let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene_env_cube {
                self.cube_cache
                    .get(&cube_cache_key(env))
                    .and_then(|gc| gc.cube_uv_view.as_ref())
                    .unwrap_or(&self.default_cube_uv_view)
            } else {
                &self.default_cube_uv_view
            };
            let env_cube_view: &wgpu::TextureView = if let Some(rt_id) = scene.environment_cube_rt {
                self.cube_rt_view_cache
                    .get(&rt_id)
                    .unwrap_or(&self.default_env_cube.view)
            } else if let Some(env) = &scene_env_cube {
                &self.cube_cache[&cube_cache_key(env)].view
            } else {
                &self.default_env_cube.view
            };
            // 4a) Each light's shadow on the glass, for the pass below, which
            //     has no room for the maps (`fs_glass_shadow`). It also lays
            //     down the glass depth that pass tests `LessEqual` against.
            let glass_shadow_view = &self.ss_glass_shadow.as_ref().unwrap().1;
            {
                let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers glass shadow pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: glass_shadow_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                sp.set_bind_group(0, &self.frame_bind_group, &[]);
                sp.set_bind_group(3, &self.env_bind_group, &[]);
                for (i, d) in draws.iter().enumerate() {
                    if !d.in_frustum || d.topology != Topology::Refract || refract_group[i] != group {
                        continue;
                    }
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        &self.pipeline_glass_shadow,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(
                        &self.device,
                        &pipe,
                        &crate::materials::RenderState::DEFAULT,
                    );
                    sp.set_pipeline(&pipe);
                    let gm = &self.geom_cache[&d.key].mesh;
                    sp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    sp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    sp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if let Some(ib) = &gm.index_buffer {
                        sp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        sp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                    } else {
                        sp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                    }
                }
            }
            // Built by hand rather than through `env_bind_group!`: this group
            // has the capture where the macro puts the shadow maps, and the
            // shadow maps are exactly what was dropped to make room for it.
            let glass_env_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers glass env bg"),
                layout: &self.env_bgl_glass,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(env_cube_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(cube_uv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&ssc.all_mips_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(ss_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(ss_back_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: self.env_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::TextureView(glass_shadow_view),
                    },
                ],
            });
            // 4) Glass surface pass (Less + depth-write, opaque composite).
            let glass_pipe = if linear_framebuffer {
                &self.pipeline_ss_glass_f16
            } else {
                &self.pipeline_ss_glass
            };
            let mut gp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers ss glass pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: layer_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            gp.set_pipeline(glass_pipe);
            gp.set_bind_group(0, &self.frame_bind_group, &[]);
            gp.set_bind_group(3, &glass_env_bg, &[]);
            for (i, d) in draws.iter().enumerate() {
                if !d.in_frustum {
                    continue;
                }
                if d.topology != Topology::Refract || refract_group[i] != group {
                    continue;
                }
                let pipe = variant_pipeline(
                    &mut self.state_pipelines,
                    &self.map_variants,
                    &self.device,
                    &self.main_shader,
                    glass_pipe,
                    d.map_variant,
                );
                let pipe = self.state_pipelines.variant(
                    &self.device,
                    &pipe,
                    &crate::materials::RenderState::DEFAULT,
                );
                gp.set_pipeline(&pipe);
                let gm = &self.geom_cache[&d.key].mesh;
                gp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                gp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    gp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                } else {
                    gp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                }
            }
            // Custom shaders that asked for the capture draw here too, with the
            // real ss-colour/ss-depth at group(3) — the whole point of the flag.
            for (i, d) in draws.iter().enumerate() {
                let Topology::Custom(h, tr, true, sd) = d.topology else {
                    continue;
                };
                if !last_group {
                    continue;
                }
                let Some(bg) = &user_bgs[i] else { continue };
                let Some(pipe) =
                    self.custom_pipelines
                        .get(&(h, linear_framebuffer, tr, true, sd))
                else {
                    continue;
                };
                let pipe = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, pipe, d.map_variant);
                let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                gp.set_pipeline(&pipe);
                gp.set_bind_group(1, bg, &[]);
                gp.set_bind_group(
                    2,
                    custom_morph_bgs[i].as_ref().unwrap_or(&self.custom_tex_bind_group),
                    &[],
                );
                let gm = &self.geom_cache[&d.key].mesh;
                gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                let sdd = &shader_data[d.shader_idx];
                let instances = if sdd.instanced {
                    gp.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                    gp.set_vertex_buffer(2, instance_buffers[sdd.custom_buf].slice(..));
                    0..d.instance_count
                } else {
                    0..1
                };
                if let Some(ib) = &gm.index_buffer {
                    gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    gp.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                } else {
                    gp.draw(draw_span(d.range, gm.vertex_count), instances);
                }
            }

            drop(gp); // end glass pass before the overlay pass reuses the encoder
            }
            // OIT surfaces in front of all the glass, over it.
            oit_phase!(refract_groups, target_view, false);

            // Overlay pass: objects opted out of glass capture draw here, over the
            // glass, depth-tested against it (so they stay crisp and un-refracted).
            if draws.iter().any(|d| !d.refract_capture) {
                let mut op = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers glass overlay pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                op.set_bind_group(0, &self.frame_bind_group, &[]);
                op.set_bind_group(3, &self.env_bind_group, &[]);
                let mut last: Option<(
                    Topology,
                    crate::materials::RenderState,
                    Option<super::material_maps::MapVariant>,
                    u32,
                )> = None;
                // Group 3 is the scene's environment unless a material has its own.
                let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                for (i, d) in draws.iter().enumerate() {
                    // A custom draw on a trimmed-environment backend (WebGL2) binds
                    // the custom pipeline's own group 3; the full environment its
                    // standard counterpart binds does not match that pipeline layout.
                    let custom_env = self.custom_env_bind_group.is_some()
                        && matches!(d.topology, Topology::Custom(..));
                    if (d.env_group, custom_env) != env_bound {
                        op.set_bind_group(
                            3,
                            if custom_env {
                                self.custom_env_bind_group.as_ref().unwrap()
                            } else {
                                d.env_group
                                    .and_then(|k| self.material_env_groups.get(&k))
                                    .map_or(&self.env_bind_group, |(_, g)| g)
                            },
                            &[],
                        );
                        env_bound = (d.env_group, custom_env);
                    }
                    if !d.in_frustum {
                        continue;
                    }
                    if d.refract_capture {
                        continue;
                    }
                    let pipe = match (linear_framebuffer, d.topology) {
                        (false, Topology::Triangle) => &self.pipeline_tri,
                        (false, Topology::TriangleAlpha) => &self.pipeline_tri_alpha,
                        (false, Topology::TriangleNoCull) => &self.pipeline_tri_nocull,
                        (false, Topology::Line) => &self.pipeline_line,
                        (false, Topology::Sprite) => &self.pipeline_sprite,
                        (false, Topology::Point) => &self.pipeline_point,
                        (false, Topology::PointAlpha) => &self.pipeline_point_alpha,
                        (false, Topology::WideLine) => &self.pipeline_wide_line,
                        (false, Topology::WideLineAlpha) => &self.pipeline_wide_line_alpha,
                        (false, Topology::Instanced) => &self.pipeline_instanced,
                        (false, Topology::InstancedAlpha) => &self.pipeline_instanced_alpha,
                        (true, Topology::Triangle) => &self.pipeline_tri_f16,
                        (true, Topology::TriangleAlpha) => &self.pipeline_tri_alpha_f16,
                        (true, Topology::TriangleNoCull) => &self.pipeline_tri_nocull_f16,
                        (true, Topology::Line) => &self.pipeline_line_f16,
                        (true, Topology::Sprite) => &self.pipeline_sprite_f16,
                        (true, Topology::Point) => &self.pipeline_point_f16,
                        (true, Topology::PointAlpha) => &self.pipeline_point_alpha_f16,
                        (true, Topology::WideLine) => &self.pipeline_wide_line_f16,
                        (true, Topology::WideLineAlpha) => &self.pipeline_wide_line_alpha_f16,
                        (true, Topology::Instanced) => &self.pipeline_instanced_f16,
                        (true, Topology::InstancedAlpha) => &self.pipeline_instanced_alpha_f16,
                        _ => continue, // skinned/custom/etc not supported as overlay
                    };
                    if last != Some((d.topology, d.state, d.map_variant, d.side)) {
                        let pipe = variant_pipeline(&mut self.state_pipelines, &self.map_variants, &self.device, &self.main_shader, pipe, d.map_variant);
                        let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                        let pipe = match side_cull(d.topology.is_faces(), d.side) {
                            Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                            None => pipe,
                        };
                        op.set_pipeline(&pipe);
                        last = Some((d.topology, d.state, d.map_variant, d.side));
                    }
                    let gm = if d.topology == Topology::Sprite {
                        &self.sprite_quad
                    } else {
                        &self.geom_cache[&d.key].mesh
                    };
                    op.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    op.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    // Wide lines: a capsule (six corners) per segment in the range —
                    // an index span, or a vertex span, two to a segment — the segment's
                    // start through slot 0 and its end a vertex further through slot 1.
                    if d.topology.is_wide_lines() {
                        let Some(buf) = &gm.segment_buffer else { continue };
                        let span = draw_span(d.range, gm.segment_vertices);
                        let segs = span.start / 2..span.end / 2;
                        if segs.is_empty() {
                            continue;
                        }
                        op.set_vertex_buffer(0, buf.slice(..));
                        op.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                        op.draw(0..6, segs);
                        continue;
                    }
                    op.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if d.topology.is_points() {
                        op.draw(0..6, draw_span(d.range, gm.vertex_count));
                        continue;
                    }
                    let inst = if d.topology.is_instanced() {
                        op.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        d.instance_count
                    } else {
                        1
                    };
                    if let Some(ib) = &gm.index_buffer {
                        op.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        op.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..inst);
                    } else {
                        op.draw(draw_span(d.range, gm.vertex_count), 0..inst);
                    }
                }
            }
        }

        // --- Temporal anti-aliasing: reproject + blend the rendered frame with
        // the accumulated history, then blit the result back to the target. ---
        if taa_active {
            let (depth_tex, color_tex) = taa_src.unwrap();
            let (w, h) = self.depth_size;
            // ACCUMULATE IN FLOAT, NOT IN THE FRAMEBUFFER'S FORMAT.
            //
            // A running mean folds each new sample in at `1/(n+1)`, and in an
            // eight-bit history that is below the quantum long before the mean
            // has settled: by the thirty-second sample the increment is a
            // 255th of the difference, which rounds to nothing, and every
            // sample after it is discarded. The mean stops improving while the
            // caller keeps paying for samples. Sixteen float bits cost one more
            // byte a pixel at output size and make every sample count.
            //
            // Only when converging. Ordinary TAA blends at a fixed 0.9 and
            // never gets near the quantum, and it is the interactive path,
            // where the bandwidth is worth more than the precision.
            let hist_format = if self.taa_converge {
                wgpu::TextureFormat::Rgba16Float
            } else {
                ss_format
            };
            self.ensure_taa_targets(w, h, hist_format);
            let first = self.taa_frame == 0;
            // Camera movement, carried for a caller that can distinguish its
            // own pinned geometry. This pass cannot: see TAA_SHADER.
            let cam = camera.position();
            let d = [
                cam.x - self.taa_prev_cam[0],
                cam.y - self.taa_prev_cam[1],
                cam.z - self.taa_prev_cam[2],
            ];
            let taa_u = TaaUniforms {
                inv_view_proj: taa_inv_vp,
                prev_view_proj: self.taa_prev_view_proj,
                // Converging: `n/(n+1)` makes the history the mean of every
                // sample so far. Otherwise the usual fixed feedback.
                params: [
                    w as f32,
                    h as f32,
                    if self.taa_converge {
                        self.taa_frame as f32 / (self.taa_frame as f32 + 1.0)
                    } else {
                        0.9
                    },
                    if first { 1.0 } else { 0.0 },
                ],
                // w: 0 feedback, 1 converging over sRGB-encoded bytes, 2
                // converging over linear light. The converge branch needs to
                // know which, because a running mean of encoded bytes is a mean
                // over a curve: see TAA_SHADER.
                cam_delta: [
                    d[0],
                    d[1],
                    d[2],
                    match (self.taa_converge, shader_outputs_linear) {
                        (false, _) => 0.0,
                        (true, false) => 1.0,
                        (true, true) => 2.0,
                    },
                ],
                flags: [if self.state_pipelines.reversed() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            };
            self.queue
                .write_buffer(&self.taa_uniform, 0, bytemuck::bytes_of(&taa_u));
            let cur_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            // frame N reads what N-1 wrote, writes the other buffer.
            let even = self.taa_frame.is_multiple_of(2);
            if self.taa_bind_cache.is_none() {
                let (_, _, _, a_view, _, b_view) = self.taa_history.as_ref().unwrap();
                let mk_resolve = |read: &wgpu::TextureView| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers taa bg"),
                        layout: &self.taa_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&cur_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(read),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&self.taa_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.taa_uniform.as_entire_binding(),
                            },
                        ],
                    })
                };
                let mk_blit = |write: &wgpu::TextureView| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers taa blit bg"),
                        layout: &self.ss_blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(write),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                            },
                        ],
                    })
                };
                // Index by `taa_frame % 2`: even frames read b and write a.
                self.taa_bind_cache = Some(TaaBindCache {
                    resolve: [mk_resolve(b_view), mk_resolve(a_view)],
                    blit: [mk_blit(a_view), mk_blit(b_view)],
                });
            }
            let parity = usize::from(!even);
            let (a_view, b_view) = {
                let (_, _, _, av, _, bv) = self.taa_history.as_ref().unwrap();
                (av, bv)
            };
            let write_view = if even { a_view } else { b_view };
            let resolve_bg = &self.taa_bind_cache.as_ref().unwrap().resolve[parity];
            // The resolve draws into the history, so its target format is the
            // history's — float whenever converging, whatever the framebuffer.
            let taa_pipe = if linear_framebuffer || self.taa_converge {
                &self.pipeline_taa_f16
            } else {
                &self.pipeline_taa
            };
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers taa resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write_view,
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
                rp.set_pipeline(taa_pipe);
                rp.set_bind_group(0, resolve_bg, &[]);
                rp.draw(0..3, 0..1);
            }
            // Blit the resolved history back to the frame target — unless this
            // is an intermediate sample of an accumulation run, in which case
            // the mean is in the history buffer and nothing will read the
            // target before the next sample overwrites it anyway.
            if self.taa_blit {
            let blit_bg = &self.taa_bind_cache.as_ref().unwrap().blit[parity];
            let blit_pipe = if linear_framebuffer {
                &self.pipeline_ss_blit_f16
            } else {
                &self.pipeline_ss_blit
            };
            {
                let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers taa blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
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
                bp.set_pipeline(blit_pipe);
                bp.set_bind_group(0, blit_bg, &[]);
                bp.draw(0..3, 0..1);
            }
            }
        }
        // Roll the motion-blur matrices on every frame -- but a PROBE IS NOT A
        // FRAME. An environment cube is six renders from a camera that is not
        // the picture's, and rolling on those leaves `mb_last_vp` holding a
        // 90-degree view from somewhere else. The next real frame then blurs
        // along the difference between the two, which is enormous: the vehicle
        // smears into a translucent double of itself and streaks run across the
        // whole image. It reads exactly like a transparency bug, and was chased
        // as one for a long time -- through the OIT buffers, the cube format,
        // the PMREM prefilter and the probe's brightness -- none of which it
        // ever was.
        if !self.cube_pass {
            self.mb_inv_vp = taa_inv_vp;
            self.mb_prev_vp = self.mb_last_vp;
            self.mb_last_vp = taa_unjittered_vp;
        }
        if taa_active && !self.cube_pass {
            // (separate block so the history-view borrows above have ended)
            self.taa_prev_view_proj = taa_unjittered_vp;
            {
                let c = camera.position();
                self.taa_prev_cam = [c.x, c.y, c.z];
            }
            self.taa_frame += 1;
        }

        if let Some(t) = &self.timer {
            t.resolve(&mut encoder);
        }
        if update_shadows {
            if let Some(t) = &self.shadow_timer {
                t.resolve(&mut encoder);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        // After submit, so the map is queued behind this frame's work. Never
        // blocks — see `gpu_timer`.
        if let Some(t) = &mut self.timer {
            let device = self.device.clone();
            t.poll(&device);
        }
        if let Some(t) = &mut self.shadow_timer {
            let device = self.device.clone();
            t.poll(&device);
        }
        let _ = (&self.frame_bgl,);
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
        self.shadow_timer.as_ref().map(|t| t.last_ms()).unwrap_or(0.0)
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

    fn ensure_slot_uploaded(&mut self, slot: &Option<Arc<Texture>>) {
        let Some(t) = slot else {
            return;
        };
        // Skip upload for render-target-backed textures — the view lives in
        // the RenderTarget itself.
        if t.external_rt_id.is_some() {
            return;
        }
        let key = tex_cache_key(t);
        // No liveness check any more: the key is the texture's own id, which
        // does not depend on where — or whether — its bytes are. That is what
        // lets a data-less `Texture::handle` resolve to the upload its full
        // twin made, and it closes the recycled-address hole the weak handle
        // was guarding, since an id is never handed out twice.
        if !self.tex_cache.contains_key(&key) {
            // Drop stale GPU copies for this logical texture (older upload_seq).
            self.tex_cache.retain(|k, _| k.0 != t.id);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
            if t.is_handle() {
                // A handle carries no pixels: it can only ever resolve to an
                // upload its full twin already made. Reaching here means that
                // upload is missing, and what gets created instead is a texture
                // of the right size with nothing in it -- which samples black
                // and raises no error anywhere.
                log::warn!(
                    "uploading a data-less handle: id {} {}x{} {:?} -- this samples black",
                    t.id,
                    t.width,
                    t.height,
                    t.format
                );
                if std::env::var("TEXWARN").is_ok() {
                    eprintln!(
                        "   WARN data-less handle uploaded: id {} {}x{} {:?}",
                        t.id, t.width, t.height, t.format
                    );
                }
            }
            // Downsampling is a texture read and a write, so let the GPU do it
            // rather than computing the whole chain serially on this thread and
            // uploading it level by level.
            let mipgen = self
                .mipgen
                .get_or_insert_with(|| MipGenerator::new(&self.device));
            let gt = GpuTexture::upload_with_mipgen(&self.device, &self.queue, t, mipgen);
            self.tex_cache.insert(key, gt);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
    }

    fn view_for<'a>(
        &'a self,
        slot: &Option<Arc<Texture>>,
        default: &'a GpuTexture,
    ) -> &'a wgpu::TextureView {
        match slot {
            Some(t) => {
                // If this Texture is RT-backed, sample from the RT's color view.
                if let Some(rt_id) = t.external_rt_id {
                    if let Some(view) = self.rt_view_cache.get(&rt_id) {
                        return view;
                    }
                }
                &self.tex_cache[&tex_cache_key(t)].view
            }
            None => &default.view,
        }
    }

    /// Borrow an RT's sample-side view by id. Caller must have previously
    /// `register_render_target`'d this id. Used by post-fx to bind the input.
    pub fn rt_view_cache_get(&self, rt_id: u32) -> &wgpu::TextureView {
        self.rt_view_cache
            .get(&rt_id)
            .expect("rt id not registered")
    }

    /// Register a CubeRenderTarget so scene.environment_cube_rt can sample
    /// from its 6-face cube view as an env map.
    pub fn register_cube_render_target(&mut self, rt_id: u32, rt: &Arc<super::CubeRenderTarget>) {
        let sample_format = rt.format.add_srgb_suffix();
        let view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers cube rt sample view"),
            format: Some(sample_format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            base_mip_level: 0,
            mip_level_count: Some(1),
            aspect: wgpu::TextureAspect::All,
            usage: None,
        });
        self.cube_rt_view_cache.insert(rt_id, view);
    }

    /// Register a RenderTarget so RT-backed Textures can sample from its color view.
    /// Creates a sample-side view using the sRGB-variant of the storage format
    /// (when applicable) so the hardware does sRGB→linear decode on read,
    /// matching three.js's WebGL behavior where RT textures auto-decode.
    pub fn register_render_target(&mut self, rt_id: u32, rt: &Arc<super::RenderTarget>) {
        let sample_format = rt_sample_format(rt.format);
        let view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt sample view"),
            format: Some(sample_format),
            ..Default::default()
        });
        self.rt_view_cache.insert(rt_id, view);
        self.tex_bg_cache.clear();
        self.variant_bg_cache.clear();
        let linear_view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt linear sample view"),
            format: Some(rt.format),
            ..Default::default()
        });
        self.rt_linear_view_cache.insert(rt_id, linear_view);
        let depth_view = rt.depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt depth sample view"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.rt_depth_view_cache.insert(rt_id, depth_view);
        self.rt_registry.insert(rt_id, Arc::clone(rt));
    }

    fn build_tex_bg(
        &self,
        slots: &MaterialTextureSlots,
        morph: Option<&wgpu::TextureView>,
        uv_sets: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let morph = morph.unwrap_or(&self.default_morph.1);
        let uv_sets = uv_sets.unwrap_or(&self.default_uv_sets.1);
        let albedo = self.view_for(&slots.map, &self.default_white_srgb);
        let normal = self.view_for(&slots.normal_map, &self.default_normal);
        let roughness = self.view_for(&slots.roughness_map, &self.default_white_linear);
        let metalness = self.view_for(&slots.metalness_map, &self.default_white_linear);
        let ao = self.view_for(&slots.ao_map, &self.default_white_linear);
        let emissive = self.view_for(&slots.emissive_map, &self.default_white_srgb);
        // Matcap and cloud cover share a binding. WebGPU guarantees only 16
        // sampled textures per fragment stage and we are at that ceiling, so a
        // seventeenth is not available — but the two can never both be set: a
        // matcap belongs to `MatcapMaterial`, a cloud deck to a Standard or
        // Physical surface.
        let matcap = match &slots.cloud_shadow_map {
            Some(_) => self.view_for(&slots.cloud_shadow_map, &self.default_transparent),
            None => self.view_for(&slots.matcap_map, &self.default_white_srgb),
        };

        // One sampler for every map of the material, from the colour map's
        // filter and wrap (or the first map present) — see
        // `MaterialTextureSlots::sampler_source`.
        let chosen = slots.sampler_source();
        let sampler = match chosen {
            Some(t) => self.pick_sampler(t),
            None => &self.samplers[SAMPLER_DEFAULT],
        };

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers tex bg"),
            layout: &self.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(metalness),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(emissive),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(matcap),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(morph),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(uv_sets),
                },
            ],
        })
    }

    /// Select one of the 4 pre-built samplers based on a Texture's filter/wrap
    /// fields. Mirrors three.js: NearestFilter→nearest, LinearFilter→linear;
    /// RepeatWrapping→repeat, ClampToEdgeWrapping→clamp.
    /// Build material shader variant `v` — its module and group-2 layout — if
    /// it is not built yet.
    fn ensure_map_variant(&mut self, v: super::material_maps::MapVariant) {
        use super::material_maps as mm;
        if self.map_variants.contains_key(&v) {
            return;
        }
        let source = super::shader::shader_source_with_maps(&mm::prelude(v));
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers material variant shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let b = mm::bindings(v);
        let mut entries = vec![uv_sets_entry(), wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        }];
        for (role, t, smp) in &b.maps {
            let visibility = if role.vertex() {
                wgpu::ShaderStages::VERTEX
            } else {
                wgpu::ShaderStages::FRAGMENT
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: *t,
                visibility,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: *smp,
                visibility,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        if let Some((t, smp)) = b.env {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: t,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: smp,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: b.xforms,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers material variant bgl"),
            entries: &entries,
        });
        self.map_variants.insert(v, (module, bgl));
    }

    /// Group 2 for a draw on material variant `v`: each map with its own
    /// sampler, and the block of their uv transforms.
    fn build_variant_tex_bg(
        &self,
        slots: &MaterialTextureSlots,
        morph: Option<&wgpu::TextureView>,
        uv_sets: Option<&wgpu::TextureView>,
        v: super::material_maps::MapVariant,
        uniforms: [[f32; 4]; 6],
    ) -> wgpu::BindGroup {
        use super::material_maps as mm;
        let b = mm::bindings(v);
        let textures: Vec<&std::sync::Arc<Texture>> = mm::role_textures(slots)
            .into_iter()
            .filter(|(r, _)| v.contains(*r))
            .filter_map(|(_, t)| t)
            .collect();
        let mut rows: Vec<[f32; 4]> = Vec::with_capacity(6 + 2 * textures.len().max(1));
        rows.extend(uniforms);
        // Row 0's w: the uv set the map reads (three.js `Texture.channel`).
        for t in &textures {
            let [r0, r1] = t.uv_transform_rows();
            rows.push([r0[0], r0[1], r0[2], t.channel.min(3) as f32]);
            rows.push([r1[0], r1[1], r1[2], 0.0]);
        }
        if textures.is_empty() {
            rows.extend([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
        }
        let xforms = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers material map transforms"),
                contents: bytemuck::cast_slice(&rows),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let morph = morph.unwrap_or(&self.default_morph.1);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(morph),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(uv_sets.unwrap_or(&self.default_uv_sets.1)),
            },
        ];
        for ((_, t, smp), tex) in b.maps.iter().zip(&textures) {
            let slot = Some((*tex).clone());
            entries.push(wgpu::BindGroupEntry {
                binding: *t,
                resource: wgpu::BindingResource::TextureView(self.view_for(&slot, &self.default_white_linear)),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: *smp,
                resource: wgpu::BindingResource::Sampler(self.pick_sampler(tex)),
            });
        }
        if let (Some((t, smp)), Some(cube)) = (b.env, slots.env_map.as_ref()) {
            entries.push(wgpu::BindGroupEntry {
                binding: t,
                resource: wgpu::BindingResource::TextureView(&self.cube_cache[&cube_cache_key(cube)].view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: smp,
                resource: wgpu::BindingResource::Sampler(&self.samplers[cube_sampler_index(cube)]),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: b.xforms,
            resource: xforms.as_entire_binding(),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers material variant bg"),
            layout: &self.map_variants[&v].1,
            entries: &entries,
        })
    }

    /// three.js `WebGLCubeUVMaps`: a cube texture lighting a Standard or
    /// Physical material is read through its PMREM — prefiltered here once,
    /// and kept — unless it already carries one (`PMREMGenerator` output).
    fn prefiltered(&mut self, cube: &std::sync::Arc<CubeTexture>) -> std::sync::Arc<CubeTexture> {
        if cube.cube_uv_atlas.is_some() {
            return cube.clone();
        }
        self.pmrem_cache
            .entry(cube_cache_key(cube))
            .or_insert_with(|| crate::PmremGenerator::from_cube(cube.clone()))
            .clone()
    }

    /// [`prefiltered`](Self::prefiltered) once it has run for `cube`.
    fn prefiltered_cached(&self, cube: &std::sync::Arc<CubeTexture>) -> std::sync::Arc<CubeTexture> {
        self.pmrem_cache.get(&cube_cache_key(cube)).cloned().unwrap_or_else(|| cube.clone())
    }

    fn pick_sampler<'a>(&'a self, t: &Texture) -> &'a wgpu::Sampler {
        &self.samplers[sampler_index(t)]
    }

    fn make_default_tex_bg(&self) -> wgpu::BindGroup {
        self.build_tex_bg(&MaterialTextureSlots::default(), None, None)
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
) -> wgpu::RenderPipeline {
    let alpha = code == 1;
    let vertex_buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 48, shader_location: 8, format: wgpu::VertexFormat::Float32x4 },
        ],
    })];
    let mut ds = depth_stencil(!alpha, wgpu::CompareFunction::LessEqual);
    ds.bias = wgpu::DepthBiasState {
        constant: units,
        slope_scale: factor,
        clamp: 0.0,
    };
    pipes.create(device, &wgpu::RenderPipelineDescriptor {
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
            if code == 2 { None } else { Some(wgpu::Face::Back) },
        ),
        depth_stencil: Some(ds),
        multisample: wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
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
            .flat_map(|b| b.chunks_exact(4))
            .map(|c| super::gpu_texture::f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    };
    #[cfg(threers_no_rect_area_light)]
    let half: Vec<u16> = vec![0; 64 * 64 * 4 * 2];
    let size = wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 2 };
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
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(64 * 8), rows_per_image: Some(64) },
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
            wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(capture) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
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
        let module = custom_shader_module_source("@fragment\nfn fs_main() {}\n", None, true, false, 1);
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
