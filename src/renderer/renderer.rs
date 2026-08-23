//! wgpu scene renderer: pipelines, resource caches, shadows, and post-processing.
//!
//! Walks a [`Scene`](crate::Scene), batches draw calls by material and topology,
//! and writes to a swapchain view or [`super::RenderTarget`]. Post-fx passes
//! share WGSL in [`super::shader`] (`POSTFX_SHADER`, effect kinds 0–30).

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;

use super::gpu_mesh::{geom_cache_key, GpuMesh};
use super::gpu_texture::{
    cube_cache_key, tex_cache_key, GpuCubeTexture, GpuTexture, MipGenerator, TexCacheKey,
};
use super::shader::{
    MAX_DIR_LIGHTS, MAX_HEMI_LIGHTS, MAX_POINT_LIGHTS, MAX_RECT_LIGHTS, MAX_SPOT_LIGHTS,
    SHADER_SOURCE,
};
use crate::cameras::Camera;
use crate::core::{BufferGeometry, ObjectKind};
use crate::lights::Light;
use crate::materials::{Material, MaterialKind, MaterialTextureSlots};
use crate::math::{Matrix3, Matrix4, Vector3};
use crate::scene::Scene;
use crate::textures::{CubeTexture, Texture, TextureFormat};

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view: [f32; 16],
    projection: [f32; 16],
    view_proj: [f32; 16],
    shadow_vp: [f32; 16],
    spot_shadow_vp: [f32; 16],
    cube_face_vp: [f32; 16],
    camera_position: [f32; 4],
    ambient: [f32; 4],
    light_counts: [u32; 4],
    /// x: dir shadow on, y: dir bias, z: spot shadow on, w: spot bias
    shadow_params: [f32; 4],
    /// xyz: point shadow caster world position, w: radius (0 = disabled)
    point_shadow_pos: [f32; 4],
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
    env_map_params: [f32; 4],
    dir_lights: [DirLightGpu; MAX_DIR_LIGHTS],
    point_lights: [PointLightGpu; MAX_POINT_LIGHTS],
    spot_lights: [SpotLightGpu; MAX_SPOT_LIGHTS],
    hemi_lights: [HemiLightGpu; MAX_HEMI_LIGHTS],
    rect_lights: [RectLightGpu; MAX_RECT_LIGHTS],
    /// x: rect-area light count. yzw reserved. `light_counts` was already full.
    light_counts2: [u32; 4],
    /// One texel of each shadow map, as a fraction of the map: x directional,
    /// y spot. The PCF kernels size themselves from this rather than from a
    /// constant compiled into the shader — the map is allocated on the Rust
    /// side and can change, and a WGSL constant cannot be told.
    shadow_texel: [f32; 4],
}

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            view: [0.0; 16],
            projection: [0.0; 16],
            view_proj: [0.0; 16],
            shadow_vp: [0.0; 16],
            spot_shadow_vp: [0.0; 16],
            cube_face_vp: [0.0; 16],
            camera_position: [0.0; 4],
            ambient: [0.0; 4],
            light_counts: [0; 4],
            shadow_params: [0.0; 4],
            point_shadow_pos: [0.0; 4],
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
            shadow_texel: [0.0; 4],
        }
    }
}

/// Build the environment `@group(3)` bind group. The layout (13 bindings: env
/// cube + samplers, three shadow maps + comparison samplers, the CubeUV atlas,
/// and the screen-space glass color/depth/back-depth) is fixed here in one place,
/// so the four construction sites (default, shadow-dummies, per-frame, glass)
/// just pass the 13 views/samplers in binding order and a new binding is added
/// once. Texture vs. sampler wrapping is encoded by the arg position.
macro_rules! env_bind_group {
    ($device:expr, $bgl:expr, $label:expr,
     $t0:expr, $s1:expr, $t2:expr, $s3:expr, $t4:expr, $s5:expr,
     $t6:expr, $s7:expr, $t8:expr, $t9:expr, $s10:expr, $t11:expr, $t12:expr $(,)?) => {
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $bgl,
            entries: &[
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
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView($t9),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler($s10),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView($t11),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView($t12),
                },
            ],
        })
    };
}

/// Uniform for the TAA resolve pass — see `TAA_SHADER`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct TaaUniforms {
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    /// x: width, y: height, z: history weight, w: first-frame flag.
    params: [f32; 4],
    /// xyz: how far the camera moved since the previous frame. w: the distance
    /// past which a pixel is treated as sky and reprojected by rotation alone.
    cam_delta: [f32; 4],
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
    /// x: ao_intensity, y/z: normal_scale, w: toon_steps (bitcast u32 in WGSL)
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
const FLAG_IRIDESCENCE_MAP: u32 = 1024;
const FLAG_DISPLACEMENT_MAP: u32 = 2048;
const FLAG_CLOUD_SHADOW: u32 = 4096;

struct CachedGpuMesh {
    version: u32,
    mesh: GpuMesh,
}

/// Camera parameters forwarded into SSAO / SSR post-fx uniforms.
#[derive(Clone)]
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
}

impl ToneMapping {
    /// The mode index the WGSL side switches on.
    fn shader_mode(self) -> f32 {
        match self {
            ToneMapping::None => 0.0,
            ToneMapping::Linear => 1.0,
            ToneMapping::AcesFilmic => 2.0,
        }
    }
}

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
    // Weighted-blended OIT: accumulation pipeline (MRT) + resolve (sRGB + f16).
    pipeline_oit: wgpu::RenderPipeline,
    pipeline_oit_resolve: wgpu::RenderPipeline,
    pipeline_oit_resolve_f16: wgpu::RenderPipeline,
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
    pipeline_sprite: wgpu::RenderPipeline,
    // Hardware-MSAA variants of the opaque forward pipelines (surface path).
    pipeline_tri_msaa: wgpu::RenderPipeline,
    pipeline_tri_alpha_msaa: wgpu::RenderPipeline,
    pipeline_tri_nocull_msaa: wgpu::RenderPipeline,
    pipeline_tri_sky_msaa: wgpu::RenderPipeline,
    pipeline_tri_wire_msaa: wgpu::RenderPipeline,
    pipeline_line_msaa: wgpu::RenderPipeline,
    pipeline_point_msaa: wgpu::RenderPipeline,
    /// MSAA sample count for the surface pass: 1 = off, else `MSAA_SAMPLES`.
    msaa: u32,
    tone_mapping: ToneMapping,
    exposure: f32,
    /// Lazily-created multisampled color + depth targets, keyed by size.
    msaa_targets: Option<(u32, u32, wgpu::TextureView, wgpu::TextureView)>,
    /// Half-float color-target variants for EffectComposer / postfx RTs.
    pipeline_tri_f16: wgpu::RenderPipeline,
    pipeline_tri_alpha_f16: wgpu::RenderPipeline,
    pipeline_tri_nocull_f16: wgpu::RenderPipeline,
    pipeline_tri_sky_f16: wgpu::RenderPipeline,
    pipeline_tri_wire_f16: wgpu::RenderPipeline,
    pipeline_line_f16: wgpu::RenderPipeline,
    pipeline_point_f16: wgpu::RenderPipeline,
    pipeline_sprite_f16: wgpu::RenderPipeline,
    pipeline_instanced_f16: wgpu::RenderPipeline,
    pipeline_skinned_f16: wgpu::RenderPipeline,
    pipeline_shadow: wgpu::RenderPipeline,
    pipeline_shadow_spot: wgpu::RenderPipeline,
    pipeline_shadow_point: wgpu::RenderPipeline,
    pipeline_instanced: wgpu::RenderPipeline,
    pipeline_skinned: wgpu::RenderPipeline,
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
    /// Uploaded by SSAOPass — 32 hemisphere samples (vec4 stride).
    ssao_kernel_buffer: wgpu::Buffer,
    ssao_noise_tex: wgpu::Texture,
    ssao_noise_view: wgpu::TextureView,
    /// 64×64 R8 displacement map for GlitchPass (DigitalGlitch tDisp).
    glitch_disp_tex: wgpu::Texture,
    glitch_disp_view: wgpu::TextureView,
    /// Full-screen R8 snow map for GlitchPass (precomputed in JS for GLSL parity).
    glitch_snow_tex: wgpu::Texture,
    glitch_snow_view: wgpu::TextureView,
    /// CPU-uploaded RGBA8 for WebGL postfx fallback blit to canvas/RT.
    rgba_upload_tex: wgpu::Texture,
    rgba_upload_view: wgpu::TextureView,
    /// Alpha-blended 2D overlay pass, built on the first caption draw so
    /// projects that never draw one allocate nothing.
    caption_pass: Option<super::caption_pass::CaptionPass>,
    mesh_skinned_bgl: wgpu::BindGroupLayout,
    sprite_quad: GpuMesh,
    frame_bgl: wgpu::BindGroupLayout,
    mesh_bgl: wgpu::BindGroupLayout,
    tex_bgl: wgpu::BindGroupLayout,
    env_bgl: wgpu::BindGroupLayout,
    /// Bind group layout for custom-shader draws: mesh uniform (binding 0) plus
    /// `ShaderMaterial` user data (binding 1 uniform + bindings 2-4 storage), all
    /// in @group(1) to stay within WebGPU's 4-bind-group limit.
    custom_mesh_bgl: wgpu::BindGroupLayout,
    /// Pipeline layout for custom-shader materials ([frame, custom_mesh, tex, env]).
    custom_pipeline_layout: wgpu::PipelineLayout,
    /// Lazily-compiled custom-shader pipelines, keyed by (fragment-source hash,
    /// is-half-float-target, is-transparent).
    custom_pipelines: HashMap<(u64, bool, bool, bool, u32), wgpu::RenderPipeline>,
    depth_view: wgpu::TextureView,
    /// The renderer's own depth texture (COPY_SRC) for the direct-`render` path.
    depth_texture: wgpu::Texture,
    /// 1×1 placeholder depth texture for postfx binding 6 when hw depth is unused.
    hw_depth_tex: wgpu::Texture,
    depth_size: (u32, u32),

    geom_cache: HashMap<*const BufferGeometry, CachedGpuMesh>,
    skin_attr_cache: HashMap<*const BufferGeometry, wgpu::Buffer>,
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
    cube_cache: HashMap<*const CubeTexture, GpuCubeTexture>,
    env_bind_group: wgpu::BindGroup,
    shadow_env_bind_group: wgpu::BindGroup,
    env_sampler: wgpu::Sampler,
    default_env_cube: GpuCubeTexture,
    default_cube_uv_view: wgpu::TextureView,
    cube_face_buffers: [wgpu::Buffer; 6],
    cube_face_bind_groups: [wgpu::BindGroup; 6],
    env_cached_key: Option<*const CubeTexture>,
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
    shadow_view: wgpu::TextureView,
    /// Side of the directional shadow map currently allocated.
    shadow_size: u32,
    /// True while rendering one face of an environment cube. Screen-space
    /// passes that assume the frame's dimensions are skipped.
    cube_pass: bool,
    shadow_sampler: wgpu::Sampler,
    spot_shadow_view: wgpu::TextureView,
    spot_shadow_sampler: wgpu::Sampler,
    #[allow(dead_code)]
    point_shadow_cube_view: wgpu::TextureView,
    point_shadow_face_views: [wgpu::TextureView; 6],
    #[allow(dead_code)]
    point_shadow_sampler: wgpu::Sampler,

    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    per_mesh: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup)>, // uniform, mesh bg, tex bg

    sampler_linear: wgpu::Sampler,
    sampler_nearest_clamp: wgpu::Sampler,
    sampler_linear_repeat: wgpu::Sampler,
    sampler_nearest_repeat: wgpu::Sampler,

    // Default fallbacks (1x1 textures used when a material slot is empty).
    default_white_srgb: GpuTexture,
    default_white_linear: GpuTexture,
    default_normal: GpuTexture, // (0.5, 0.5, 1.0, 1.0) — straight up tangent normal
    /// Also the empty displacement slot: sampling black moves nothing, so the
    /// vertex stage needs no branch.
    default_black: GpuTexture,
    /// The empty cloud-shadow slot. `default_black` is opaque, and a cover map
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
    /// Depth-only pipeline rendering glass back faces (cull front) into ss_back_depth.
    pipeline_ss_back_depth: wgpu::RenderPipeline,
    /// 1×1 placeholder depth bound at env-group binding 11/12 when no glass draws.
    ss_depth_placeholder_view: wgpu::TextureView,
    /// Fullscreen "sample one texture → write" pipeline (mip downsample + blit),
    /// one per framebuffer format.
    pipeline_ss_blit: wgpu::RenderPipeline,
    pipeline_ss_blit_f16: wgpu::RenderPipeline,
    /// Glass surface pipelines (screen-space refraction/reflection).
    pipeline_ss_glass: wgpu::RenderPipeline,
    pipeline_ss_glass_f16: wgpu::RenderPipeline,
    /// BGL for the blit/downsample pipeline (texture + sampler).
    ss_blit_bgl: wgpu::BindGroupLayout,

    // --- Temporal anti-aliasing (opt-in via `set_taa`) ---
    taa_enabled: bool,
    taa_frame: u32,
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
    #[allow(dead_code)]
    texture: wgpu::Texture,
    /// View over all mips — bound to the glass shader for LOD sampling.
    all_mips_view: wgpu::TextureView,
    /// One single-mip view per level — render targets for the downsample chain.
    mip_render_views: Vec<wgpu::TextureView>,
    /// One single-mip sample view per level — sources for the downsample chain.
    mip_sample_views: Vec<wgpu::TextureView>,
    mips: u32,
    format: wgpu::TextureFormat,
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Hardware MSAA sample count used when antialiasing is enabled for the
/// direct-to-surface opaque forward pass. 4× is universally supported.
const MSAA_SAMPLES: u32 = 4;

// ---- ShaderMaterial (custom-shader) support ----------------------------------

/// Cache key for a custom shader = a hash of its fragment source.
fn custom_shader_hash(src: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

/// Assemble a full WGSL module for a custom shader: the built-in preamble
/// (everything before the built-in `fs_main` — structs, `vs_main`, bindings for
/// groups 0-3, and all helper functions), then the `@group(4)` user bindings,
/// then the user's fragment.
fn custom_shader_module_source(fragment: &str) -> String {
    let preamble = SHADER_SOURCE
        .split("@fragment")
        .next()
        .unwrap_or(SHADER_SOURCE);
    // User data lives in @group(1) alongside the mesh uniform (binding 0), to
    // stay within WebGPU's 4-bind-group limit.
    let user_bindings = "\n\
struct ThreersUserData { data: array<vec4<f32>, 16>, };\n\
@group(1) @binding(1) var<uniform> u_data: ThreersUserData;\n\
@group(1) @binding(2) var<storage, read> u_s0: array<f32>;\n\
@group(1) @binding(3) var<storage, read> u_s1: array<f32>;\n\
@group(1) @binding(4) var<storage, read> u_s2: array<f32>;\n\
@group(1) @binding(5) var u_tex0: texture_2d<f32>;\n\
@group(1) @binding(6) var u_tex1: texture_2d<f32>;\n\
@group(1) @binding(7) var u_tex2: texture_2d<f32>;\n\
@group(1) @binding(8) var u_tex3: texture_2d<f32>;\n\
@group(1) @binding(9) var u_samp: sampler;\n";
    format!("{preamble}{user_bindings}\n{fragment}\n")
}

/// Compile a custom-shader pipeline (alpha-blended, no depth write — treated as
/// transparent for correct compositing).
fn build_custom_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    fragment: &str,
    format: wgpu::TextureFormat,
    transparent: bool,
    screen_space: bool,
    side: u32,
) -> wgpu::RenderPipeline {
    let source = custom_shader_module_source(fragment);
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
    let vertex_buffers = vertex_buffers.map(Some);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("threers custom pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
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
        depth_stencil: Some(depth_stencil(
            screen_space || !transparent,
            wgpu::CompareFunction::Less,
        )),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Build the custom @group(1) bind group: the mesh uniform (binding 0, shared
/// with the standard path) plus the `ShaderMaterial` user uniform (binding 1)
/// and three storage arrays (bindings 2-4). Returns the bind group plus the new
/// buffers it owns (kept alive by the caller until the render pass is submitted).
#[allow(clippy::too_many_arguments)]
fn build_user_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    mesh_uniform: &wgpu::Buffer,
    data: &[[f32; 4]; 16],
    storage: &[Vec<f32>; 3],
    textures: &[std::sync::Arc<wgpu::TextureView>],
    placeholder: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> (wgpu::BindGroup, Vec<wgpu::Buffer>) {
    use wgpu::util::DeviceExt;
    let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
    let mut buffers = vec![
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers user uniform"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::UNIFORM,
        }),
    ];
    for s in storage.iter() {
        buffers.push(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threers user storage"),
                contents: bytemuck::cast_slice(s),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
    }
    let tex =
        |i: usize| -> &wgpu::TextureView { textures.get(i).map(|t| &**t).unwrap_or(placeholder) };
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("threers custom mesh+user bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: mesh_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffers[3].as_entire_binding(),
            },
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
        ],
    });
    (bg, buffers)
}

impl Renderer {
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
            self.geom_cache
                .insert(key, CachedGpuMesh { version: ver, mesh });
        }
    }

    /// (Re)create the OIT accumulation + revealage targets at `w×h` and their
    /// resolve bind group. No-op if already the right size.
    /// (Re)create the mipmapped opaque-scene capture used by the screen-space
    /// glass pass. Sized `w×h` in `format` (must match the framebuffer format so
    /// the opaque pipelines can render into mip 0). The mip chain feeds the
    /// roughness→LOD blur of the refracted/reflected image.
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let all_mips_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
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

    pub fn set_taa(&mut self, enabled: bool) {
        if !enabled {
            self.taa_frame = 0;
            self.taa_history = None;
        }
        self.taa_enabled = enabled;
    }

    /// Reset TAA accumulation (call on a hard cut / scene change to avoid ghosting).
    pub fn reset_taa(&mut self) {
        self.taa_frame = 0;
    }

    fn ensure_taa_targets(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) {
        let (w, h) = (w.max(1), h.max(1));
        if let Some((cw, ch, ..)) = &self.taa_history {
            if *cw == w && *ch == h {
                return;
            }
        }
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
        let (reveal, reveal_view) = mk(wgpu::TextureFormat::R16Float, "threers oit reveal");
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
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
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

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers frame bgl"),
            entries: &[uniform_entry(0)],
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 8 fragment textures (albedo, normal, roughness, metalness, ao,
        // emissive, matcap, iridescence) + 1 sampler + a displacement map that
        // is sampled in the *vertex* stage.
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
                tex_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    // The displacement sample needs the same sampler, from the
                    // vertex stage.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let env_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers env bgl"),
            entries: &[
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
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
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

        // ShaderMaterial data rides in @group(1) alongside the mesh uniform
        // (WebGPU's default maxBindGroups is 4, so we can't add a 5th group):
        //   binding 0 = mesh uniform (as usual), binding 1 = user uniform block,
        //   bindings 2..4 = read-only storage arrays.
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
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
        let custom_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let custom_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers custom mesh+user bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                // User textures + their sampler. Unbound slots get a 1x1 white
                // placeholder, so a shader that ignores them costs nothing.
                custom_tex_entry(5),
                custom_tex_entry(6),
                custom_tex_entry(7),
                custom_tex_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let custom_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers custom (shader material) layout"),
                bind_group_layouts: &[Some(&frame_bgl), Some(&custom_mesh_bgl), Some(&tex_bgl), Some(&env_bgl)],
                immediate_size: 0,
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

        let make_pipeline = |fmt: wgpu::TextureFormat,
                             topology: wgpu::PrimitiveTopology,
                             cull: Option<wgpu::Face>,
                             label: &str| {
            let alpha = label.contains("alpha");
            // Note: PolygonMode::Line needs the POLYGON_MODE_LINE feature, which
            // is generally unavailable in browser WebGPU. The wireframe pipeline
            // instead uses LineList primitives over the triangle indices (fed as
            // edge indices at draw time), so every pipeline stays Fill.
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                // transparent surface still composite correctly.
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Single-layer "glass" pipelines. `depth_only` (the prepass) writes the
        // nearest glass depth with no color; the color pass alpha-blends only the
        // fragments at that nearest depth (LessEqual) and doesn't write depth.
        let make_glass_pipeline = |fmt: wgpu::TextureFormat, depth_only: bool, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: if depth_only {
                            wgpu::ColorWrites::empty()
                        } else {
                            wgpu::ColorWrites::ALL
                        },
                    })],
                    compilation_options: Default::default(),
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
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let f16_format = wgpu::TextureFormat::Rgba16Float;
        let pipeline_glass_depth =
            make_glass_pipeline(color_format, true, "threers glass depth pipeline");
        let pipeline_glass_color =
            make_glass_pipeline(color_format, false, "threers glass color pipeline");
        let pipeline_glass_depth_f16 =
            make_glass_pipeline(f16_format, true, "threers glass depth f16 pipeline");
        let pipeline_glass_color_f16 =
            make_glass_pipeline(f16_format, false, "threers glass color f16 pipeline");

        // Screen-space refraction glass: opaque single-surface pass (writes depth,
        // Less test, no blend — it composites the refracted/reflected background
        // itself), sampling the captured ss-color from env-group binding 9.
        let make_ss_glass_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_ss_glass = make_ss_glass_pipeline(color_format, "threers ss glass pipeline");
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
        let pipeline_ss_back_depth =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                sample_type: if depth {
                    wgpu::TextureSampleType::Depth
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
        let pipeline_taa = make_taa(color_format, "threers taa pipeline");
        let pipeline_taa_f16 = make_taa(f16_format, "threers taa f16 pipeline");

        // --- Weighted-blended OIT ---
        // Accumulation pass: fs_oit writes to accum (additive) + revealage
        // (multiplicative), depth-tested against opaque but not writing depth.
        let oit_accum_format = wgpu::TextureFormat::Rgba16Float;
        let oit_reveal_format = wgpu::TextureFormat::R16Float;
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
        let pipeline_oit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers oit accum pipeline"),
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
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
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
        let make_resolve = |fmt: wgpu::TextureFormat, label: &str| {
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        let pipeline_oit_resolve = make_resolve(color_format, "threers oit resolve pipeline");
        let pipeline_oit_resolve_f16 = make_resolve(f16_format, "threers oit resolve f16 pipeline");
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
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        // PolygonMode::Line draws the triangle outline — used for `wireframe:true` materials.
        // wgpu requires the NON_FILL_POLYGON_MODE feature for Line/Point polygon modes on
        // some backends. If unsupported, this falls back to a no-op (renders nothing extra),
        // but Chromium's WebGPU/SwiftShader path tends to allow it.
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
        let pipeline_point = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::PointList,
            None,
            "threers point pipeline",
        );

        // --- Hardware-MSAA variants of the opaque forward pipelines (surface
        // path only). Identical to the ones above but with `count = MSAA_SAMPLES`;
        // used when antialiasing is enabled and the scene has no glass/OIT/refract
        // /instanced/skinned/custom draws (those fall back to the 1× path). ---
        let make_pipeline_ms = |fmt: wgpu::TextureFormat,
                                topology: wgpu::PrimitiveTopology,
                                cull: Option<wgpu::Face>,
                                label: &str| {
            let alpha = label.contains("alpha");
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let make_sky_pipeline_ms = |fmt: wgpu::TextureFormat, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        let pipeline_point_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::PointList,
            None,
            "threers point msaa",
        );
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
        let pipeline_point_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::PointList,
            None,
            "threers point f16 pipeline",
        );

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
                array_stride: 64,
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
        let pipeline_skinned = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers skinned pipeline"),
            layout: Some(&pipeline_layout_skinned),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skinned"),
                buffers: &vertex_buffers_skinned,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_skinned_f16 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers skinned f16 pipeline"),
            layout: Some(&pipeline_layout_skinned),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_skinned"),
                buffers: &vertex_buffers_skinned,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: f16_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_instanced = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers instanced pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_instanced"),
                buffers: &vertex_buffers_instanced,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_instanced_f16 =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers instanced f16 pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_instanced"),
                    buffers: &vertex_buffers_instanced,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: f16_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let pipeline_sprite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        let pipeline_sprite_f16 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        // Depth-only pipeline used to render the scene from each shadow-casting
        // light's POV. Reuses the per-vertex layout + per-mesh uniforms.
        let make_shadow_pipeline = |entry: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: None,
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
        let pipeline_shadow = make_shadow_pipeline("vs_shadow", "threers shadow pipeline");
        let pipeline_shadow_spot =
            make_shadow_pipeline("vs_shadow_spot", "threers spot shadow pipeline");
        let pipeline_shadow_point =
            make_shadow_pipeline("vs_shadow_point", "threers point shadow pipeline");

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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let glitch_snow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers glitch snow"),
            size: wgpu::Extent3d {
                width: 800,
                height: 600,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let glitch_snow_view = glitch_snow_tex.create_view(&wgpu::TextureViewDescriptor::default());
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
        // 1×1 depth bound at env-group binding 11 when no glass draws.
        let ss_depth_placeholder_view =
            hw_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers frame uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers frame bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        // One uniform buffer per cube face, because a point light's six shadow
        // passes each need a different view-projection and they are all encoded
        // before anything runs. Writing one buffer six times inside an encoder
        // does not do that: `queue.write_buffer` calls are applied before *any*
        // submitted command, so all six faces would render with the last one.
        let cube_face_buffers: [wgpu::Buffer; 6] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("threers cube face uniform {i}")),
                size: std::mem::size_of::<FrameUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cube_face_bind_groups: [wgpu::BindGroup; 6] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers cube face bg"),
                layout: &frame_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cube_face_buffers[i].as_entire_binding(),
                }],
            })
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers linear sampler"),
            // three.js's DataTexture default is ClampToEdge — matching here
            // prevents UV-1.0-corner samples from wrapping back to UV 0.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
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
        let sampler_nearest_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers nearest-clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
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
        let sampler_nearest_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers nearest-repeat sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let default_white_srgb = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8UnormSrgb),
        );
        let default_white_linear = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8Unorm),
        );
        let default_normal = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([128, 128, 255, 255], TextureFormat::Rgba8Unorm),
        );
        let default_black = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([0, 0, 0, 255], TextureFormat::Rgba8Unorm),
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
        const SHADOW_SIZE: u32 = 4096;
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers shadow depth"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let spot_shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers spot shadow depth"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let spot_shadow_view = spot_shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let spot_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers spot shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Point shadow cubemap depth (6 layers × 512×512 Depth32Float).
        const POINT_SHADOW_SIZE: u32 = 512;
        let point_shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers point shadow depth cube"),
            size: wgpu::Extent3d {
                width: POINT_SHADOW_SIZE,
                height: POINT_SHADOW_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let point_shadow_cube_view = point_shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers point shadow cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let mk_face = |i: u32| {
            point_shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers point shadow face"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let point_shadow_face_views: [wgpu::TextureView; 6] = [
            mk_face(0),
            mk_face(1),
            mk_face(2),
            mk_face(3),
            mk_face(4),
            mk_face(5),
        ];
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

        let env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers env bg",
            &default_env_cube.view,
            &env_sampler,
            &shadow_view,
            &shadow_sampler,
            &spot_shadow_view,
            &spot_shadow_sampler,
            &point_shadow_cube_view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &default_cube_uv_view,
            &ss_glass_sampler,
            &ss_depth_placeholder_view,
            &ss_depth_placeholder_view,
        );

        // Dummy 1x1 depth textures used by `shadow_env_bind_group` so the
        // shadow passes can satisfy the pipeline layout without binding the
        // textures they're currently writing to (WebGPU forbids simultaneous
        // RenderAttachment + TextureBinding usage of the same resource).
        let dummy_depth_2d = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers dummy depth 2d"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_2d_view =
            dummy_depth_2d.create_view(&wgpu::TextureViewDescriptor::default());
        let dummy_depth_cube = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers dummy depth cube"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_depth_cube_view = dummy_depth_cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers dummy depth cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let shadow_env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers shadow env bg (dummies)",
            &default_env_cube.view,
            &env_sampler,
            &dummy_depth_2d_view,
            &shadow_sampler,
            &dummy_depth_2d_view,
            &spot_shadow_sampler,
            &dummy_depth_cube_view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &default_cube_uv_view,
            &ss_glass_sampler,
            &ss_depth_placeholder_view,
            &ss_depth_placeholder_view,
        );

        Self {
            device,
            queue,
            pipeline_glass_depth,
            pipeline_glass_color,
            pipeline_glass_depth_f16,
            pipeline_glass_color_f16,
            pipeline_oit,
            pipeline_oit_resolve,
            pipeline_oit_resolve_f16,
            oit_resolve_bgl,
            oit_targets: None,
            ss_color: None,
            ss_glass_sampler,
            ss_depth: None,
            ss_back_depth: None,
            pipeline_ss_back_depth,
            ss_depth_placeholder_view,
            pipeline_ss_blit,
            pipeline_ss_blit_f16,
            pipeline_ss_glass,
            pipeline_ss_glass_f16,
            ss_blit_bgl,
            taa_enabled: false,
            taa_frame: 0,
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
            pipeline_sprite,
            pipeline_tri_msaa,
            pipeline_tri_alpha_msaa,
            pipeline_tri_nocull_msaa,
            pipeline_tri_sky_msaa,
            pipeline_tri_wire_msaa,
            pipeline_line_msaa,
            pipeline_point_msaa,
            msaa: 1,
            // Off by default: enabling tone mapping would silently re-grade
            // every existing scene and the parity renders along with them.
            tone_mapping: ToneMapping::None,
            exposure: 1.0,
            msaa_targets: None,
            pipeline_tri_f16,
            pipeline_tri_alpha_f16,
            pipeline_tri_nocull_f16,
            pipeline_tri_sky_f16,
            pipeline_tri_wire_f16,
            pipeline_line_f16,
            pipeline_point_f16,
            pipeline_sprite_f16,
            pipeline_instanced_f16,
            pipeline_skinned_f16,
            pipeline_shadow,
            pipeline_shadow_spot,
            pipeline_shadow_point,
            pipeline_instanced,
            pipeline_skinned,
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
            ssao_kernel_buffer,
            ssao_noise_tex,
            ssao_noise_view,
            glitch_disp_tex,
            glitch_disp_view,
            glitch_snow_tex,
            glitch_snow_view,
            rgba_upload_tex,
            rgba_upload_view,
            caption_pass: None,
            mesh_skinned_bgl,
            sprite_quad,
            frame_bgl,
            mesh_bgl,
            tex_bgl,
            env_bgl,
            custom_mesh_bgl,
            custom_pipeline_layout,
            custom_pipelines: HashMap::new(),
            depth_view,
            depth_texture,
            hw_depth_tex,
            depth_size: (width, height),
            geom_cache: HashMap::new(),
            skin_attr_cache: HashMap::new(),
            tex_cache: HashMap::new(),
            mipgen: None,
            rt_view_cache: HashMap::new(),
            rt_linear_view_cache: HashMap::new(),
            rt_registry: HashMap::new(),
            rt_depth_view_cache: HashMap::new(),
            cube_rt_view_cache: HashMap::new(),
            cube_cache: HashMap::new(),
            env_bind_group,
            shadow_env_bind_group,
            env_sampler,
            default_env_cube,
            default_cube_uv_view,
            cube_face_buffers,
            cube_face_bind_groups,
            env_cached_key: None,
            env_dirty: true,
            env_cached_rt_id: None,
            shadow_view,
            shadow_size: SHADOW_SIZE,
            cube_pass: false,
            shadow_sampler,
            spot_shadow_view,
            spot_shadow_sampler,
            point_shadow_cube_view,
            point_shadow_face_views,
            point_shadow_sampler,
            frame_buffer,
            frame_bind_group,
            per_mesh: Vec::new(),
            sampler_linear,
            sampler_nearest_clamp,
            sampler_linear_repeat,
            sampler_nearest_repeat,
            default_white_srgb,
            default_white_linear,
            default_normal,
            default_black,
            default_transparent,
            color_format,
        }
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
            self.msaa_targets = None; // recreated on next MSAA pass at the new size
        }
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
        }
    }
    pub fn msaa(&self) -> u32 {
        self.msaa
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

    /// Ensure the multisampled color+depth targets exist at `w×h` for the MSAA pass.
    fn ensure_msaa_targets(&mut self, w: u32, h: u32) {
        if matches!(self.msaa_targets, Some((cw, ch, ..)) if cw == w && ch == h) {
            return;
        }
        let mk = |label, format| {
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        // wgpu keeps a texture alive as long as one of its views exists, so we can
        // store only the views.
        let cv = mk("threers msaa color", self.color_format).create_view(&Default::default());
        let dv = mk("threers msaa depth", DEPTH_FORMAT).create_view(&Default::default());
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
        let use_hw_depth = kind == 11 && depth_rt_id != 0 && depth_rt_id == normal_rt_id;
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
        let flip_y = if output_rt_id == 0 {
            if kind == 0 {
                !kind0_byte_copy && params2[1] > 0.5
            } else if matches!(kind, 4 | 5 | 25 | 26) {
                // Halftone/glitch: pixel-space sampling on same-orientation RT.
                false
            } else {
                true
            }
        } else {
            false
        };
        match normal_owned {
            Some(n) => {
                let normal_bind = normal_linear.as_ref().unwrap_or(&n);
                self.apply_postfx_ex(
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
                    flip_y,
                );
                self.rt_view_cache.insert(normal_rt_id, n);
            }
            None => {
                let fallback_normal = self
                    .default_normal
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let normal_bind = normal_linear.as_ref().unwrap_or(&fallback_normal);
                self.apply_postfx_ex(
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
                    flip_y,
                );
            }
        }
        if let Some(v) = hw_depth_local {
            self.rt_depth_view_cache.insert(normal_rt_id, v);
        }
        self.rt_view_cache.insert(input_rt_id, input_view);
        if let Some(d) = depth_owned {
            self.rt_view_cache.insert(depth_rt_id, d);
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
            !matches!(kind, 4 | 5 | 25 | 26),
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
        flip_y: bool,
    ) {
        let linear_output = output_format == wgpu::TextureFormat::Rgba16Float;
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
                if kind == 5 || kind == 25 || kind == 26 {
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
                if flip_y { 1.0 } else { 0.0 },
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
                        if kind == 5 || kind == 25 || kind == 26 {
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
                    resource: wgpu::BindingResource::TextureView(if kind == 26 {
                        &self.glitch_snow_view
                    } else if kind == 5 || kind == 25 {
                        &self.glitch_disp_view
                    } else if kind == 3 && params3[3] > 0.5 {
                        &self.glitch_snow_view
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

    /// Upload GLSL-baked dotscreen pattern (R32Float; binding 8 when kind=3 params3.w>0.5).
    pub fn set_dotscreen_pattern(&mut self, data: &[f32], width: u32, height: u32) {
        self.set_glitch_snow(data, width, height);
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
        if self.rgba_upload_tex.width() != w || self.rgba_upload_tex.height() != h {
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
                format: wgpu::TextureFormat::Rgba8Unorm,
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
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 4],
            w,
            h,
            false,
            PostFxCamera::default(),
            false,
            output_format,
            true,
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
        let pass = self
            .caption_pass
            .get_or_insert_with(|| super::caption_pass::CaptionPass::new(&self.device));
        pass.draw(
            &self.device,
            &self.queue,
            data,
            width,
            height,
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
            target,
            format,
        );
    }

    /// Upload full-screen glitch snow map (R32Float, point-sampled).
    pub fn set_glitch_snow(&mut self, data: &[f32], width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if self.glitch_snow_tex.width() != w || self.glitch_snow_tex.height() != h {
            self.glitch_snow_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers glitch snow"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.glitch_snow_view = self
                .glitch_snow_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.glitch_snow_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload 64×64 glitch displacement heightmap (R32Float; manual bilinear in WGSL).
    pub fn set_glitch_disp(&mut self, data: &[f32], size: u32) {
        let sz = size.max(1);
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
    fn ensure_shadow_size(&mut self, requested: u32) {
        let size = requested.clamp(256, 8192).next_power_of_two();
        if size == self.shadow_size {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers shadow depth"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.shadow_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.shadow_size = size;
        // The environment bind group holds the old view. Invalidating the cache
        // key is what makes the next frame rebuild it; without this the shader
        // would keep sampling a texture nothing renders into any more.
        self.env_cached_key = None;
        self.env_cached_rt_id = None;
        self.env_dirty = true;
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
        // The target owns the live depth this pass — pass it as the copy source
        // for the screen-space glass ray march.
        self.render_inner(
            scene,
            camera,
            &target.color_view,
            linear_fb,
            Some(&target.depth_texture),
            Some(&target.color_texture),
        );
        self.depth_view = orig_depth;
        self.depth_size = (orig_w, orig_h);
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
        color_src: Option<&wgpu::Texture>,
    ) {
        scene.update_world();

        // TAA active only when we have a sampleable color+depth target.
        // NOT during a probe. A cube face is 48 pixels and is not the picture:
        // running temporal AA over it reprojects the frame's history through
        // the probe's camera and writes the result back, so the next real frame
        // blends against six tiny views of the scene. On screen that reads as
        // the whole vehicle going half transparent — which is what it looked
        // like, and why the cause was not obvious from the symptom.
        let taa_active =
            self.taa_enabled && !self.cube_pass && color_src.is_some() && depth_src.is_some();

        let view_m = camera.view_matrix();
        let mut proj_m = camera.projection_matrix();
        // Un-jittered view-projection is kept for TAA reprojection.
        let unjittered = proj_m.multiply(&view_m);
        if taa_active {
            // Sub-pixel Halton jitter in NDC (± half a pixel).
            let jx = (halton(self.taa_frame + 1, 2) - 0.5) * 2.0 / self.depth_size.0 as f32;
            let jy = (halton(self.taa_frame + 1, 3) - 0.5) * 2.0 / self.depth_size.1 as f32;
            proj_m.elements[8] += jx;
            proj_m.elements[9] += jy;
        }
        let view_proj = proj_m.multiply(&view_m).elements;
        let taa_inv_vp = unjittered.invert().elements;
        let taa_unjittered_vp = unjittered.elements;

        // tone_mapping_exposure[2] is the "IBL on" flag — the PBR shader's
        // env-map branch is gated on it. Enable whenever the scene has any
        // environment source (CPU cube texture or cube render target).
        let has_env = scene.environment.is_some() || scene.environment_cube_rt.is_some();
        let (cam_near, cam_far) = camera.near_far();
        let mut frame_u = FrameUniforms {
            view: view_m.elements,
            projection: proj_m.elements,
            view_proj,
            camera_position: [
                camera.position().x,
                camera.position().y,
                camera.position().z,
                0.0,
            ],
            fog_color: [scene.fog.color.r, scene.fog.color.g, scene.fog.color.b, 1.0],
            fog_params: [
                scene.fog.near,
                scene.fog.far,
                scene.fog.density,
                scene.fog.mode as f32,
            ],
            tone_mapping_exposure: [
                self.tone_mapping.shader_mode(),
                self.exposure,
                if has_env { 1.0 } else { 0.0 },
                if linear_framebuffer { 1.0 } else { 0.0 },
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
        let mut shadow_caster: Option<(Vector3, crate::lights::ShadowSettings)> = None;
        let mut spot_caster: Option<(Vector3, Vector3, f32, crate::lights::ShadowSettings)> = None;
        let mut point_caster: Option<(Vector3, f32, crate::lights::ShadowSettings)> = None;

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Topology {
            Triangle,
            TriangleAlpha,
            TriangleWire,
            TriangleNoCull,
            Sky,
            Line,
            #[allow(dead_code)]
            Point,
            Sprite,
            Instanced,
            Skinned,
            /// Single-layer glass depth prepass (writes depth, no color).
            GlassDepth,
            /// Single-layer glass color pass (LessEqual, alpha blend, no depth write).
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
            Custom(u64, bool, bool, u32),
        }
        struct DrawMesh {
            key: *const BufferGeometry,
            topology: Topology,
            /// Z in view space, used to back-to-front sort transparent draws.
            view_z: f32,
            render_order: i32,
            cast_shadow: bool,
            refract_capture: bool,
            alpha_test: f32,
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
            toon_steps: u32,
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
            /// Index into `instance_buffers` for Topology::Instanced.
            instance_buf_idx: usize,
            instance_count: u32,
            /// Index into `skin_data` for Topology::Skinned.
            skin_idx: usize,
            /// Index into `shader_data` for Topology::Custom.
            shader_idx: usize,
        }
        let mut draws: Vec<DrawMesh> = Vec::new();
        // Per-custom-shader-material draw data (parallel to Topology::Custom).
        struct ShaderDrawData {
            fragment: String,
            data: [[f32; 4]; 16],
            storage: [Vec<f32>; 3],
            textures: Vec<std::sync::Arc<wgpu::TextureView>>,
        }
        let mut shader_data: Vec<ShaderDrawData> = Vec::new();
        let mut instance_buffers: Vec<wgpu::Buffer> = Vec::new();
        struct SkinData {
            bone_buf: wgpu::Buffer,
            bg: Option<wgpu::BindGroup>,
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
            let topology = if topology == Topology::Triangle {
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
                    m.iridescence_thickness,
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
            // three.js drives every slot from one texture matrix, taken from
            // the primary map. Ours is the offset/repeat pair, which is the
            // part that matters for atlasing and tiling. It is applied once, in
            // the vertex stage, so it covers every slot — a material cannot
            // window its normal map and leave its albedo alone.
            let uv_transform = {
                let slots = mat.texture_slots();
                let t = slots.map.as_ref().or(slots.matcap_map.as_ref());
                match t {
                    Some(t) => [t.offset.x, t.offset.y, t.repeat.x, t.repeat.y],
                    None => [0.0, 0.0, 1.0, 1.0],
                }
            };
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
            DrawMesh {
                key,
                topology,
                view_z,
                render_order: obj.render_order,
                cast_shadow: obj.cast_shadow,
                refract_capture: obj.refract_capture,
                alpha_test: mat.alpha_test(),
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
                specular: [s.r, s.g, s.b, 0.0],
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
                toon_steps: mat.toon_steps(),
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
                instance_buf_idx: usize::MAX,
                instance_count: 1,
                skin_idx: usize::MAX,
                shader_idx: usize::MAX,
            }
        };

        let cam_layers = camera.layers();
        let root = scene.root;
        scene.arena.traverse_visible(root, &mut |_id, obj| {
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
                        // First cast_shadow dir light becomes the shadow caster.
                        if l.cast_shadow && n_dir == 0 && shadow_caster.is_none() {
                            shadow_caster = Some((dir, l.shadow));
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
                        if l.cast_shadow && n_point == 0 && point_caster.is_none() {
                            let radius = if l.distance > 0.0 { l.distance } else { 50.0 };
                            point_caster = Some((p, radius, l.shadow));
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
                        if l.cast_shadow && n_spot == 0 && spot_caster.is_none() {
                            spot_caster = Some((p, dir, l.angle, l.shadow));
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
                        let pos = obj.world_position();
                        let right = transform_direction(
                            &obj.matrix_world,
                            Vector3::new(l.width * 0.5, 0.0, 0.0),
                        );
                        let up = transform_direction(
                            &obj.matrix_world,
                            Vector3::new(0.0, l.height * 0.5, 0.0),
                        );
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
                    let mut d = make_draw(&mesh.geometry, &mesh.material, obj, Topology::Triangle);
                    // Custom-shader material: route to a per-source pipeline and
                    // stash its user data for the @group(4) bind group.
                    if let Material::Shader(sm) = &*mesh.material {
                        let hash = custom_shader_hash(&sm.fragment);
                        let transparent = sm.transparent || sm.opacity < 1.0;
                        d.topology = Topology::Custom(hash, transparent, sm.screen_space, sm.side);
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
                            data: sm.data_slots(),
                            storage: [pad(&sm.storage0), pad(&sm.storage1), pad(&sm.storage2)],
                            textures: sm.textures.clone(),
                        });
                    }
                    // Route transparent PhysicalMaterials to single-layer glass
                    // (depth prepass + color) or weighted-blended OIT.
                    if d.topology == Topology::TriangleAlpha {
                        match mesh.material.transparency_mode() {
                            crate::materials::TransparencyMode::Glass => {
                                draws.push(make_draw(
                                    &mesh.geometry,
                                    &mesh.material,
                                    obj,
                                    Topology::GlassDepth,
                                ));
                                d.topology = Topology::GlassColor;
                            }
                            crate::materials::TransparencyMode::Oit => d.topology = Topology::Oit,
                            crate::materials::TransparencyMode::Refract => {
                                d.topology = Topology::Refract
                            }
                            crate::materials::TransparencyMode::Blend => {}
                        }
                    }
                    draws.push(d);
                }
                ObjectKind::LineSegments(ls) => {
                    self.ensure_geometry(&ls.geometry);
                    draws.push(make_draw(&ls.geometry, &ls.material, obj, Topology::Line));
                }
                ObjectKind::Points(p) => {
                    // three.js's PointsMaterial.size is a screen-space pixel
                    // size. WebGPU's PointList topology can't honor that —
                    // it always renders 1 fragment per point. We instead emit
                    // one camera-facing sprite-style draw per point, sized to
                    // match three.js's pixel-space convention.
                    let positions = p.geometry.get_attribute("position");
                    let size = match &*p.material {
                        crate::Material::Points(pm) => pm.size,
                        _ => 1.0,
                    };
                    let mat = &*p.material;
                    let nm3 = Matrix3::normal_matrix(&obj.matrix_world);
                    let c = mat.color();
                    if let Some(positions) = positions {
                        // three.js renders points as `size`-pixel squares via
                        // `gl_PointSize`. WebGPU has no per-vertex point size, so
                        // we emit a sprite quad. The sprite vertex shader scales
                        // by the model matrix's basis lengths and applies the
                        // perspective projection — so for the right at-distance
                        // size we work backwards from the desired NDC width.
                        //
                        // For a perspective camera, an object at view-z `vz` has
                        // a world-space width of `2 * vz / proj[0][0]` per unit
                        // of NDC. We want a sprite of `size` pixels wide on
                        // an 800-wide canvas (= size/400 NDC), so the model
                        // matrix's X basis length should be:
                        //   sx = (size / 400) * vz / proj[0][0]
                        let viewport_h = 600.0_f32;
                        let proj00 = proj_m.elements[0];
                        let canvas_w = 800.0_f32;
                        let _ = viewport_h;
                        let cnt = positions.count();
                        for vi in 0..cnt {
                            let px = positions.array[vi * 3];
                            let py = positions.array[vi * 3 + 1];
                            let pz = positions.array[vi * 3 + 2];
                            let m_local = crate::math::Matrix4::translation(
                                crate::math::Vector3::new(px, py, pz),
                            );
                            let mut model = obj.matrix_world.multiply(&m_local);
                            // Project the point centre to view space to get its z, then
                            // size the quad to match the requested pixel size at that depth.
                            let wp = crate::math::Vector3::new(
                                model.elements[12],
                                model.elements[13],
                                model.elements[14],
                            );
                            let vz = (view_m.elements[2] * wp.x
                                + view_m.elements[6] * wp.y
                                + view_m.elements[10] * wp.z
                                + view_m.elements[14])
                                .abs()
                                .max(0.0001);
                            let half_ndc = (size / canvas_w) * vz / proj00;
                            // Sprite vs uses `in.position * sx`; the unit quad has corners
                            // at ±0.5, so we want sx == size_in_world. Multiply by 2 because
                            // pixel size means total quad width, not half.
                            let sx = half_ndc * 2.0;
                            let mut e = model.elements;
                            for k in 0..3 {
                                e[k] *= sx;
                                e[4 + k] *= sx;
                            }
                            model.elements = e;
                            let world_pos = crate::math::Vector3::new(e[12], e[13], e[14]);
                            let svz = view_m.elements[2] * world_pos.x
                                + view_m.elements[6] * world_pos.y
                                + view_m.elements[10] * world_pos.z
                                + view_m.elements[14];
                            draws.push(DrawMesh {
                                key: std::ptr::null::<BufferGeometry>(),
                                topology: Topology::Sprite,
                                view_z: svz,
                                render_order: obj.render_order,
                                cast_shadow: false,
                                refract_capture: true,
                                alpha_test: 0.0,
                                model: model.elements,
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
                                toon_steps: 0,
                                physical: [0.0; 4],
                                physical2: [0.0; 4],
                                physical3: [0.0; 4],
                                physical4: [0.0; 4],
                                physical5: [0.0; 4],
                                displacement: [0.0; 4],
                                uv_transform: [0.0, 0.0, 1.0, 1.0],
                                atmos: [0.0; 4],
                                twilight: [0.0; 4],
                                eclipse: [0.0; 4],
                                scatter: [0.0; 4],
                                aurora: [0.0; 4],
                                shader_flags: 0,
                                kind: crate::materials::MaterialKind::Basic,
                                slots: mat.texture_slots(),
                                instance_buf_idx: usize::MAX,
                                instance_count: 1,
                                skin_idx: usize::MAX,
                                shader_idx: usize::MAX,
                            });
                        }
                    }
                }
                ObjectKind::Sprite(sprite) => {
                    // Inline construction — the sprite quad is built-in, so we
                    // can't reuse make_draw (which keys off an Arc<BufferGeometry>).
                    let mat = &*sprite.material;
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
                        cast_shadow: false,
                        refract_capture: true,
                        alpha_test: 0.0,
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
                        toon_steps: 0,
                        physical: [0.0; 4],
                        physical2: [0.0; 4],
                        physical3: [0.0; 4],
                        physical4: [0.0; 4],
                        physical5: [0.0; 4],
                        displacement: [0.0; 4],
                        uv_transform: [0.0, 0.0, 1.0, 1.0],
                        atmos: [0.0; 4],
                        twilight: [0.0; 4],
                        eclipse: [0.0; 4],
                        scatter: [0.0; 4],
                        aurora: [0.0; 4],
                        shader_flags: 0,
                        kind: mat.kind(),
                        slots: mat.texture_slots(),
                        instance_buf_idx: usize::MAX,
                        instance_count: 1,
                        skin_idx: usize::MAX,
                        shader_idx: usize::MAX,
                    });
                }
                ObjectKind::InstancedMesh(im) => {
                    self.ensure_geometry(&im.geometry);
                    if im.transforms.is_empty() {
                        return;
                    }
                    let mut matrices: Vec<f32> = Vec::with_capacity(im.transforms.len() * 16);
                    for t in &im.transforms {
                        matrices.extend_from_slice(&t.elements);
                    }
                    let buf = wgpu::util::DeviceExt::create_buffer_init(
                        &*self.device,
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("threers instance buffer"),
                            contents: bytemuck::cast_slice(&matrices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    );
                    let idx = instance_buffers.len();
                    instance_buffers.push(buf);
                    let mut d = make_draw(&im.geometry, &im.material, obj, Topology::Instanced);
                    d.instance_buf_idx = idx;
                    d.instance_count = im.transforms.len() as u32;
                    draws.push(d);
                }
                ObjectKind::SkinnedMesh(sm) => {
                    self.ensure_geometry(&sm.geometry);
                    let key = geom_cache_key(&sm.geometry);
                    // Upload joints+weights to slot-1 vertex buffer (cached per-geometry).
                    if !self.skin_attr_cache.contains_key(&key) {
                        let joints = sm.geometry.get_attribute("joint");
                        let weights = sm.geometry.get_attribute("weight");
                        let (Some(j), Some(w)) = (joints, weights) else {
                            // Geometry isn't skin-equipped; draw at rest.
                            draws.push(make_draw(
                                &sm.geometry,
                                &sm.material,
                                obj,
                                Topology::Triangle,
                            ));
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
                    let mut bone_data: Vec<f32> =
                        Vec::with_capacity(sm.skeleton.bone_matrices.len() * 16);
                    if sm.skeleton.bone_matrices.is_empty() {
                        // Identity fallback to keep storage buffer non-empty.
                        let id = crate::math::Matrix4::identity();
                        bone_data.extend_from_slice(&id.elements);
                    } else {
                        for m in &sm.skeleton.bone_matrices {
                            bone_data.extend_from_slice(&m.elements);
                        }
                    }
                    let bone_buf = wgpu::util::DeviceExt::create_buffer_init(
                        &*self.device,
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("threers bone matrices"),
                            contents: bytemuck::cast_slice(&bone_data),
                            usage: wgpu::BufferUsages::STORAGE,
                        },
                    );
                    // Defer creating the combined mesh+skin bind group until
                    // after per_mesh is populated below — we need the per-mesh
                    // uniform buffer at binding 0.
                    let idx = skin_data.len();
                    skin_data.push(SkinData { bone_buf, bg: None });
                    let mut d = make_draw(&sm.geometry, &sm.material, obj, Topology::Skinned);
                    d.skin_idx = idx;
                    draws.push(d);
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
                Topology::GlassColor => 2,
                Topology::TriangleAlpha | Topology::Sprite | Topology::Oit | Topology::Refract => 3,
                Topology::Custom(_, transparent, ..) if transparent => 3,
                _ => 0,
            }
        }
        draws.sort_by(|a, b| {
            ord_class(a.topology)
                .cmp(&ord_class(b.topology))
                .then(a.render_order.cmp(&b.render_order))
                .then_with(|| {
                    // For ordinary transparents we want farther-from-camera first.
                    // view_z is negative in front of the camera; smaller = farther.
                    if ord_class(a.topology) == 3 {
                        a.view_z
                            .partial_cmp(&b.view_z)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
        });

        frame_u.ambient = [ambient[0], ambient[1], ambient[2], 0.0];
        frame_u.light_counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        frame_u.light_counts2 = [n_rect as u32, 0, 0, 0];

        // Shadow VP: orthographic from the caster light's direction. Camera fits
        // a fixed cube centered on the world origin (matches the default
        // ShadowSettings.camera_size). For a richer integration the box would
        // track the visible meshes' AABB.
        if let Some((dir, settings)) = shadow_caster {
            // The map is allocated before any light exists to ask for a size,
            // so the first frame that names one reallocates. `map_size` used to
            // be read and discarded, which is a worse failure than not offering
            // it: the caller sets 4096, sees no error, and gets 1024.
            self.ensure_shadow_size(settings.map_size);
            frame_u.shadow_texel[0] = 1.0 / self.shadow_size as f32;
            let s = settings.camera_size;
            let light_eye = -dir.normalize() * (s * 1.5);
            let light_view = Matrix4::look_at(light_eye, Vector3::ZERO, Vector3::UP);
            let light_proj =
                Matrix4::orthographic(-s, s, s, -s, settings.camera_near, settings.camera_far);
            let light_vp = light_proj.multiply(&light_view);
            frame_u.shadow_vp = light_vp.elements;
            frame_u.shadow_params[0] = 1.0;
            frame_u.shadow_params[1] = settings.bias.max(0.0);
            // `normal_bias` was declared, documented and then never read — the
            // same failure the `map_size` note above describes. It is read now.
            frame_u.shadow_texel[1] = settings.normal_bias.max(0.0);
        }
        if let Some((pos, radius, _settings)) = point_caster {
            frame_u.point_shadow_pos = [pos.x, pos.y, pos.z, radius];
        }
        if let Some((pos, dir, angle, settings)) = spot_caster {
            let target = pos + dir.normalize();
            let light_view = Matrix4::look_at(pos, target, Vector3::UP);
            // Perspective frustum: fov = 2 * outer_angle to cover the cone.
            let fov = (angle * 2.0).clamp(0.05, std::f32::consts::PI - 0.05);
            let light_proj =
                Matrix4::perspective(fov, 1.0, settings.camera_near, settings.camera_far);
            let light_vp = light_proj.multiply(&light_view);
            frame_u.spot_shadow_vp = light_vp.elements;
            frame_u.shadow_params[2] = 1.0;
            frame_u.shadow_params[3] = settings.bias.max(0.0);
        }

        // Sync environment cubemap. Upload on first sight, rebuild bind group on change.
        let env_key = scene.environment.as_ref().map(cube_cache_key);
        let env_rt_id = scene.environment_cube_rt;
        if self.env_dirty || env_key != self.env_cached_key || env_rt_id != self.env_cached_rt_id {
            let mut env_map_params = [0.0_f32; 4];
            if let Some(env) = &scene.environment {
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
            let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene.environment {
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
            } else if let Some(env) = &scene.environment {
                &self.cube_cache[&cube_cache_key(env)].view
            } else {
                &self.default_env_cube.view
            };
            self.env_bind_group = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers env bg",
                view,
                &self.env_sampler,
                &self.shadow_view,
                &self.shadow_sampler,
                &self.spot_shadow_view,
                &self.spot_shadow_sampler,
                &self.point_shadow_cube_view,
                &self.point_shadow_sampler,
                cube_uv_view,
                &self.default_cube_uv_view,
                &self.ss_glass_sampler,
                &self.ss_depth_placeholder_view,
                &self.ss_depth_placeholder_view,
            );
            self.env_dirty = false;
            self.env_cached_key = env_key;
            self.env_cached_rt_id = env_rt_id;
        }
        if let Some(env) = &scene.environment {
            if let Some(atlas) = &env.cube_uv_atlas {
                frame_u.env_map_params = [
                    atlas.texel_width,
                    atlas.texel_height,
                    atlas.lod_max as f32,
                    1.0,
                ];
            }
        }

        // -- Ensure every texture referenced in this frame is uploaded. --
        for d in &draws {
            self.ensure_slot_uploaded(&d.slots.map);
            self.ensure_slot_uploaded(&d.slots.normal_map);
            self.ensure_slot_uploaded(&d.slots.roughness_map);
            self.ensure_slot_uploaded(&d.slots.metalness_map);
            self.ensure_slot_uploaded(&d.slots.ao_map);
            self.ensure_slot_uploaded(&d.slots.emissive_map);
            self.ensure_slot_uploaded(&d.slots.matcap_map);
            self.ensure_slot_uploaded(&d.slots.displacement_map);
            self.ensure_slot_uploaded(&d.slots.cloud_shadow_map);
        }

        // -- Ensure pool capacity. --
        while self.per_mesh.len() < draws.len() {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers mesh uniform"),
                size: std::mem::size_of::<MeshUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mesh_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers mesh bg"),
                layout: &self.mesh_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            // Texture bind group is rebuilt per-frame per-mesh below, but we
            // need a placeholder; we'll overwrite the pool entry in-place each
            // frame. Build a "default everything" tex bg for initialization.
            let tex_bg = self.make_default_tex_bg();
            self.per_mesh.push((buf, mesh_bg, tex_bg));
        }

        // -- Build combined mesh+skin bind groups now that per_mesh is grown. --
        for (i, d) in draws.iter().enumerate() {
            if d.topology == Topology::Skinned && d.skin_idx < skin_data.len() {
                let sd = &mut skin_data[d.skin_idx];
                let mesh_uniform_buf = &self.per_mesh[i].0;
                sd.bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers mesh skinned bg"),
                    layout: &self.mesh_skinned_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: mesh_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sd.bone_buf.as_entire_binding(),
                        },
                    ],
                }));
            }
        }

        // -- Upload uniforms + rebuild per-mesh tex bind groups. --
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame_u));
        let mut new_tex_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(draws.len());
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
            if d.slots.iridescence_thickness_map.is_some() {
                flags |= FLAG_IRIDESCENCE_MAP;
            }
            if d.slots.displacement_map.is_some() {
                flags |= FLAG_DISPLACEMENT_MAP;
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
                    f32::from_bits(d.toon_steps),
                ],
                params3: d.physical,
                params4: d.physical2,
                params5: d.physical3,
                params6: d.physical4,
                params7: d.physical5,
                params8: d.displacement,
                params9: d.uv_transform,
                params10: d.atmos,
                params11: d.twilight,
                params12: d.eclipse,
                params13: d.scatter,
                params14: d.aurora,
                flags: [d.kind as u32, flags, d.shader_flags, d.alpha_test.to_bits()],
            };
            self.queue
                .write_buffer(&self.per_mesh[i].0, 0, bytemuck::bytes_of(&u));

            new_tex_bgs.push(self.build_tex_bg(&d.slots));
        }
        // Apply new tex bgs (separate loop to keep borrow rules happy).
        for (i, bg) in new_tex_bgs.into_iter().enumerate() {
            self.per_mesh[i].2 = bg;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers encoder"),
            });

        // Shadow depth pre-pass: render scene from the directional light's POV.
        if shadow_caster.is_some() {
            let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
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
            spass.set_pipeline(&self.pipeline_shadow);
            spass.set_bind_group(0, &self.frame_bind_group, &[]);
            for (i, d) in draws.iter().enumerate() {
                if !d.cast_shadow {
                    continue;
                }
                if d.topology != Topology::Triangle && d.topology != Topology::TriangleNoCull {
                    continue;
                }
                let gm = &self.geom_cache[&d.key].mesh;
                spass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                spass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                spass.set_bind_group(3, &self.shadow_env_bind_group, &[]);
                spass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    spass.draw_indexed(0..gm.index_count, 0, 0..1);
                } else {
                    spass.draw(0..gm.vertex_count, 0..1);
                }
            }
        }

        // Spot shadow depth pre-pass.
        if spot_caster.is_some() {
            let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers spot shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.spot_shadow_view,
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
            spass.set_pipeline(&self.pipeline_shadow_spot);
            spass.set_bind_group(0, &self.frame_bind_group, &[]);
            for (i, d) in draws.iter().enumerate() {
                if !d.cast_shadow {
                    continue;
                }
                if d.topology != Topology::Triangle && d.topology != Topology::TriangleNoCull {
                    continue;
                }
                let gm = &self.geom_cache[&d.key].mesh;
                spass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                spass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                spass.set_bind_group(3, &self.shadow_env_bind_group, &[]);
                spass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    spass.draw_indexed(0..gm.index_count, 0, 0..1);
                } else {
                    spass.draw(0..gm.vertex_count, 0..1);
                }
            }
        }

        // Point-light cubemap depth: 6 face passes, one per cube face.
        if let Some((pos, radius, _)) = point_caster {
            self.queue.submit(Some(encoder.finish()));
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("threers point shadow encoder"),
                });
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
            for (face, &(forward, up)) in face_lookat.iter().enumerate() {
                let target = pos + forward;
                let view = Matrix4::look_at(pos, target, up);
                let vp = proj.multiply(&view);
                frame_u.cube_face_vp = vp.elements;
                // Each face gets its own buffer — see `cube_face_buffers`.
                self.queue.write_buffer(
                    &self.cube_face_buffers[face],
                    0,
                    bytemuck::bytes_of(&frame_u),
                );

                let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers point shadow face pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.point_shadow_face_views[face],
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
                spass.set_pipeline(&self.pipeline_shadow_point);
                spass.set_bind_group(0, &self.cube_face_bind_groups[face], &[]);
                for (i, d) in draws.iter().enumerate() {
                    if d.topology != Topology::Triangle && d.topology != Topology::TriangleNoCull {
                        continue;
                    }
                    let gm = &self.geom_cache[&d.key].mesh;
                    spass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    spass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                    spass.set_bind_group(3, &self.shadow_env_bind_group, &[]);
                    spass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if let Some(ib) = &gm.index_buffer {
                        spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        spass.draw_indexed(0..gm.index_count, 0, 0..1);
                    } else {
                        spass.draw(0..gm.vertex_count, 0..1);
                    }
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
        let refract_present = !self.cube_pass
            && draws.iter().any(|d| {
                d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, _))
            });
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
        for (i, d) in draws.iter().enumerate() {
            if let Topology::Custom(hash, transparent, screen_space, side) = d.topology {
                let sd = &shader_data[d.shader_idx];
                let key = (hash, linear_framebuffer, transparent, screen_space, side);
                if !self.custom_pipelines.contains_key(&key) {
                    let fmt = if linear_framebuffer {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        self.color_format
                    };
                    let pipeline = build_custom_pipeline(
                        &self.device,
                        &self.custom_pipeline_layout,
                        &sd.fragment,
                        fmt,
                        transparent,
                        screen_space,
                        side,
                    );
                    self.custom_pipelines.insert(key, pipeline);
                }
                let (bg, bufs) = build_user_bind_group(
                    &self.device,
                    &self.custom_mesh_bgl,
                    &self.per_mesh[i].0,
                    &sd.data,
                    &sd.storage,
                    &sd.textures,
                    &self.default_white_linear.view,
                    &self.sampler_linear_repeat,
                );
                _custom_keep_alive.extend(bufs);
                user_bgs[i] = Some(bg);
            }
        }

        {
            let bg_color = scene.background;
            // HalfFloat postfx RTs store linear light; sRGB canvas clears use raw bytes
            // (matches three.js gl.clearColor on an sRGB default framebuffer).
            let bg_alpha = scene.background_alpha as f64;
            let clear = if linear_framebuffer {
                wgpu::Color {
                    r: bg_color.r.powf(2.2) as f64,
                    g: bg_color.g.powf(2.2) as f64,
                    b: bg_color.b.powf(2.2) as f64,
                    a: bg_alpha,
                }
            } else {
                wgpu::Color {
                    r: bg_color.r as f64,
                    g: bg_color.g as f64,
                    b: bg_color.b as f64,
                    a: bg_alpha,
                }
            };
            // Hardware MSAA on the direct-to-surface opaque pass. Falls back to
            // single-sample for render-to-texture, linear/postfx targets, and any
            // scene using glass/OIT/refraction/instanced/skinned/custom draws.
            let use_msaa = self.msaa > 1
                && depth_src.is_none()
                && !linear_framebuffer
                && !refract_present
                && draws.iter().all(|d| {
                    matches!(
                        d.topology,
                        Topology::Triangle
                            | Topology::TriangleAlpha
                            | Topology::TriangleNoCull
                            | Topology::Sky
                            | Topology::TriangleWire
                            | Topology::Line
                            | Topology::Point
                    )
                });
            if use_msaa {
                self.ensure_msaa_targets(self.depth_size.0, self.depth_size.1);
            }
            // Screen-space glass redirects the opaque pass into the ss-color capture
            // (mip 0); MSAA renders into the multisampled color and resolves into the
            // frame view; otherwise it targets the frame view directly.
            let (main_color_view, msaa_resolve, depth_pass_view): (
                &wgpu::TextureView,
                Option<&wgpu::TextureView>,
                &wgpu::TextureView,
            ) = if use_msaa {
                let (_, _, cv, dv) = self.msaa_targets.as_ref().unwrap();
                (cv, Some(target_view), dv)
            } else if refract_present {
                (
                    &self.ss_color.as_ref().unwrap().mip_render_views[0],
                    None,
                    &self.depth_view,
                )
            } else {
                (target_view, None, &self.depth_view)
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: main_color_view,
                    depth_slice: None,
                    resolve_target: msaa_resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_pass_view,
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

            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(3, &self.env_bind_group, &[]);
            let mut last_topology: Option<Topology> = None;
            for (i, d) in draws.iter().enumerate() {
                // OIT + screen-space-refraction surfaces are drawn in dedicated
                // later passes, not here.
                if d.topology == Topology::Oit
                    || d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, _))
                {
                    continue;
                }
                // Objects opted out of glass capture draw *over* the glass (a
                // later overlay pass), so they're not refracted — skip here.
                if refract_present && !d.refract_capture {
                    continue;
                }
                if last_topology != Some(d.topology) {
                    let pipe = if use_msaa {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri_msaa,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha_msaa,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull_msaa,
                            Topology::Sky => &self.pipeline_tri_sky_msaa,
                            Topology::TriangleWire => &self.pipeline_tri_wire_msaa,
                            Topology::Line => &self.pipeline_line_msaa,
                            Topology::Point => &self.pipeline_point_msaa,
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
                            Topology::Sky => &self.pipeline_tri_sky_f16,
                            Topology::TriangleWire => &self.pipeline_tri_wire_f16,
                            Topology::Line => &self.pipeline_line_f16,
                            Topology::Point => &self.pipeline_point_f16,
                            Topology::Sprite => &self.pipeline_sprite_f16,
                            Topology::Instanced => &self.pipeline_instanced_f16,
                            Topology::Skinned => &self.pipeline_skinned_f16,
                            Topology::Oit | Topology::Refract => continue,
                            Topology::Custom(h, tr, ss, sd) => {
                                &self.custom_pipelines[&(h, true, tr, ss, sd)]
                            }
                        }
                    } else {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull,
                            Topology::GlassDepth => &self.pipeline_glass_depth,
                            Topology::GlassColor => &self.pipeline_glass_color,
                            Topology::Sky => &self.pipeline_tri_sky,
                            Topology::TriangleWire => &self.pipeline_tri_wire,
                            Topology::Line => &self.pipeline_line,
                            Topology::Point => &self.pipeline_point,
                            Topology::Sprite => &self.pipeline_sprite,
                            Topology::Instanced => &self.pipeline_instanced,
                            Topology::Skinned => &self.pipeline_skinned,
                            Topology::Oit | Topology::Refract => continue,
                            Topology::Custom(h, tr, ss, sd) => {
                                &self.custom_pipelines[&(h, false, tr, ss, sd)]
                            }
                        }
                    };
                    pass.set_pipeline(pipe);
                    last_topology = Some(d.topology);
                }
                let gm = if d.topology == Topology::Sprite {
                    &self.sprite_quad
                } else {
                    &self.geom_cache[&d.key].mesh
                };
                if d.topology == Topology::Skinned {
                    // Skinned pipeline uses the combined mesh+skin BGL at group(1).
                    if let Some(bg) = skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()) {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                    if let Some(skin_attrs) = self.skin_attr_cache.get(&d.key) {
                        pass.set_vertex_buffer(1, skin_attrs.slice(..));
                    }
                } else if let Topology::Custom(..) = d.topology {
                    // Custom-shader draws use the combined mesh+user bind group
                    // at @group(1) (mesh uniform + user data/storage).
                    if let Some(bg) = &user_bgs[i] {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                } else {
                    pass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    pass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                }
                pass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                let instance_count = if d.topology == Topology::Instanced {
                    pass.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                    d.instance_count
                } else {
                    1
                };
                // For TriangleWire we draw the wireframe line-list index buffer
                // (three.js WebGLGeometries: a,b,b,c,c,a per triangle).
                let (ib_opt, idx_count) = if d.topology == Topology::TriangleWire {
                    (gm.wire_index_buffer.as_ref(), gm.wire_index_count)
                } else {
                    (gm.index_buffer.as_ref(), gm.index_count)
                };
                if let Some(ib) = ib_opt {
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..idx_count, 0, 0..instance_count);
                } else {
                    pass.draw(0..gm.vertex_count, 0..instance_count);
                }
            }
        }

        // --- Weighted-blended OIT: accumulate the OIT surfaces, then resolve
        // (composite) over the opaque image already in `target_view`. ---
        // Not during a probe, and this one bites hard. The order-independent
        // transparency targets are sized from `depth_size`, which a cube pass
        // has set to the face's 48 pixels — so a probe RESIZES them, and the
        // next real frame resolves a 48-pixel transparency buffer stretched
        // across the whole picture. It reads as the entire vehicle going
        // translucent, which looks like an alpha bug in the materials and is
        // nothing of the kind.
        if !self.cube_pass && draws.iter().any(|d| d.topology == Topology::Oit) {
            let (tw, th) = self.depth_size;
            self.ensure_oit_targets(tw, th);
            if let Some((_, _, _, accum_view, _, reveal_view, resolve_bg)) =
                self.oit_targets.as_ref()
            {
                // 1) Accumulation pass (accum cleared to 0, revealage cleared to 1).
                {
                    let mut opass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit accum pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: accum_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: reveal_view,
                                depth_slice: None,
                                resolve_target: None,
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
                    opass.set_pipeline(&self.pipeline_oit);
                    opass.set_bind_group(0, &self.frame_bind_group, &[]);
                    opass.set_bind_group(3, &self.env_bind_group, &[]);
                    for (i, d) in draws.iter().enumerate() {
                        if d.topology != Topology::Oit {
                            continue;
                        }
                        let gm = &self.geom_cache[&d.key].mesh;
                        opass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                        opass.set_bind_group(2, &self.per_mesh[i].2, &[]);
                        opass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if let Some(ib) = &gm.index_buffer {
                            opass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            opass.draw_indexed(0..gm.index_count, 0, 0..1);
                        } else {
                            opass.draw(0..gm.vertex_count, 0..1);
                        }
                    }
                }
                // 2) Resolve over the opaque image (alpha-blended fullscreen).
                {
                    let resolve_pipe = if linear_framebuffer {
                        &self.pipeline_oit_resolve_f16
                    } else {
                        &self.pipeline_oit_resolve
                    };
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit resolve pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
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
                    rpass.set_pipeline(resolve_pipe);
                    rpass.set_bind_group(0, resolve_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }
            }
        }

        // --- Screen-space refraction glass: build the ss-color mip chain from
        // the opaque capture, blit the sharp capture to the frame target, then
        // draw the glass surface sampling it for refraction + reflection. ---
        if refract_present {
            let ssc = self.ss_color.as_ref().unwrap();
            // Copy the opaque depth into the sampleable ss_depth so the glass
            // shader can ray-march it for refraction/reflection.
            if let Some((_, _, ss_dtex, _)) = &self.ss_depth {
                let src = depth_src.unwrap_or(&self.depth_texture);
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
                            load: wgpu::LoadOp::Clear(1.0),
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
                    if d.topology != Topology::Refract {
                        continue;
                    }
                    let gm = &self.geom_cache[&d.key].mesh;
                    bp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    bp.set_bind_group(2, &self.per_mesh[i].2, &[]);
                    bp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if let Some(ib) = &gm.index_buffer {
                        bp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        bp.draw_indexed(0..gm.index_count, 0, 0..1);
                    } else {
                        bp.draw(0..gm.vertex_count, 0..1);
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
                bp.set_bind_group(0, &blit_bg, &[]);
                bp.draw(0..3, 0..1);
            }
            // 3) group(3) bind group with the real ss-color at binding 9 (env cube
            //    + shadows mirror the frame's env bind group).
            let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene.environment {
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
            } else if let Some(env) = &scene.environment {
                &self.cube_cache[&cube_cache_key(env)].view
            } else {
                &self.default_env_cube.view
            };
            let glass_env_bg = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers glass env bg",
                env_cube_view,
                &self.env_sampler,
                &self.shadow_view,
                &self.shadow_sampler,
                &self.spot_shadow_view,
                &self.spot_shadow_sampler,
                &self.point_shadow_cube_view,
                &self.point_shadow_sampler,
                cube_uv_view,
                &ssc.all_mips_view,
                &self.ss_glass_sampler,
                ss_depth_view,
                ss_back_depth_view,
            );
            // 4) Glass surface pass (Less + depth-write, opaque composite).
            let glass_pipe = if linear_framebuffer {
                &self.pipeline_ss_glass_f16
            } else {
                &self.pipeline_ss_glass
            };
            let mut gp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers ss glass pass"),
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
            gp.set_pipeline(glass_pipe);
            gp.set_bind_group(0, &self.frame_bind_group, &[]);
            gp.set_bind_group(3, &glass_env_bg, &[]);
            for (i, d) in draws.iter().enumerate() {
                if d.topology != Topology::Refract {
                    continue;
                }
                let gm = &self.geom_cache[&d.key].mesh;
                gp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                gp.set_bind_group(2, &self.per_mesh[i].2, &[]);
                gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    gp.draw_indexed(0..gm.index_count, 0, 0..1);
                } else {
                    gp.draw(0..gm.vertex_count, 0..1);
                }
            }
            // Custom shaders that asked for the capture draw here too, with the
            // real ss-colour/ss-depth at group(3) — the whole point of the flag.
            for (i, d) in draws.iter().enumerate() {
                let Topology::Custom(h, tr, true, sd) = d.topology else {
                    continue;
                };
                let Some(bg) = &user_bgs[i] else { continue };
                gp.set_pipeline(&self.custom_pipelines[&(h, linear_framebuffer, tr, true, sd)]);
                gp.set_bind_group(1, bg, &[]);
                gp.set_bind_group(2, &self.per_mesh[i].2, &[]);
                let gm = &self.geom_cache[&d.key].mesh;
                gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    gp.draw_indexed(0..gm.index_count, 0, 0..1);
                } else {
                    gp.draw(0..gm.vertex_count, 0..1);
                }
            }

            drop(gp); // end glass pass before the overlay pass reuses the encoder

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
                let mut last: Option<Topology> = None;
                for (i, d) in draws.iter().enumerate() {
                    if d.refract_capture {
                        continue;
                    }
                    let pipe = match (linear_framebuffer, d.topology) {
                        (false, Topology::Triangle) => &self.pipeline_tri,
                        (false, Topology::TriangleAlpha) => &self.pipeline_tri_alpha,
                        (false, Topology::TriangleNoCull) => &self.pipeline_tri_nocull,
                        (false, Topology::Line) => &self.pipeline_line,
                        (false, Topology::Sprite) => &self.pipeline_sprite,
                        (false, Topology::Instanced) => &self.pipeline_instanced,
                        (true, Topology::Triangle) => &self.pipeline_tri_f16,
                        (true, Topology::TriangleAlpha) => &self.pipeline_tri_alpha_f16,
                        (true, Topology::TriangleNoCull) => &self.pipeline_tri_nocull_f16,
                        (true, Topology::Line) => &self.pipeline_line_f16,
                        (true, Topology::Sprite) => &self.pipeline_sprite_f16,
                        (true, Topology::Instanced) => &self.pipeline_instanced_f16,
                        _ => continue, // skinned/custom/etc not supported as overlay
                    };
                    if last != Some(d.topology) {
                        op.set_pipeline(pipe);
                        last = Some(d.topology);
                    }
                    let gm = if d.topology == Topology::Sprite {
                        &self.sprite_quad
                    } else {
                        &self.geom_cache[&d.key].mesh
                    };
                    op.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    op.set_bind_group(2, &self.per_mesh[i].2, &[]);
                    op.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    let inst = if d.topology == Topology::Instanced {
                        op.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        d.instance_count
                    } else {
                        1
                    };
                    if let Some(ib) = &gm.index_buffer {
                        op.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        op.draw_indexed(0..gm.index_count, 0, 0..inst);
                    } else {
                        op.draw(0..gm.vertex_count, 0..inst);
                    }
                }
            }
        }

        // --- Temporal anti-aliasing: reproject + blend the rendered frame with
        // the accumulated history, then blit the result back to the target. ---
        if taa_active {
            let color_tex = color_src.unwrap();
            let depth_tex = depth_src.unwrap();
            let (w, h) = self.depth_size;
            self.ensure_taa_targets(w, h, ss_format);
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
                params: [w as f32, h as f32, 0.9, if first { 1.0 } else { 0.0 }],
                cam_delta: [d[0], d[1], d[2], 0.0],
            };
            self.queue
                .write_buffer(&self.taa_uniform, 0, bytemuck::bytes_of(&taa_u));
            let cur_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            // frame N reads what N-1 wrote, writes the other buffer.
            let (a_view, b_view) = {
                let (_, _, _, av, _, bv) = self.taa_history.as_ref().unwrap();
                (av, bv)
            };
            let even = self.taa_frame.is_multiple_of(2);
            let (read_view, write_view) = if even {
                (b_view, a_view)
            } else {
                (a_view, b_view)
            };
            let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers taa bg"),
                layout: &self.taa_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&cur_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(read_view),
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
            });
            let taa_pipe = if linear_framebuffer {
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
                rp.set_bind_group(0, &resolve_bg, &[]);
                rp.draw(0..3, 0..1);
            }
            // Blit the resolved history back to the frame target.
            let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers taa blit bg"),
                layout: &self.ss_blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(write_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                    },
                ],
            });
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
                bp.set_bind_group(0, &blit_bg, &[]);
                bp.draw(0..3, 0..1);
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

        self.queue.submit(Some(encoder.finish()));
        let _ = (&self.frame_bgl,);
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

    fn build_tex_bg(&self, slots: &MaterialTextureSlots) -> wgpu::BindGroup {
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
        let iridescence_thickness =
            self.view_for(&slots.iridescence_thickness_map, &self.default_white_linear);
        // Black default: no map means no displacement, so the vertex stage can
        // sample unconditionally and still move nothing.
        let displacement = self.view_for(&slots.displacement_map, &self.default_black);

        // Pick sampler based on the primary texture's filter/wrap. We use
        // `slots.map` as the canonical choice (matcap uses `matcap_map`).
        let chosen = slots.map.as_ref().or(slots.matcap_map.as_ref());
        let sampler = match chosen {
            Some(t) => self.pick_sampler(t),
            None => &self.sampler_linear,
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
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(displacement),
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
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(iridescence_thickness),
                },
            ],
        })
    }

    /// Select one of the 4 pre-built samplers based on a Texture's filter/wrap
    /// fields. Mirrors three.js: NearestFilter→nearest, LinearFilter→linear;
    /// RepeatWrapping→repeat, ClampToEdgeWrapping→clamp.
    fn pick_sampler<'a>(&'a self, t: &Texture) -> &'a wgpu::Sampler {
        use crate::textures::{TextureFilter, TextureWrap};
        let nearest = matches!(t.mag_filter, TextureFilter::Nearest);
        let repeat =
            matches!(t.wrap_s, TextureWrap::Repeat) || matches!(t.wrap_t, TextureWrap::Repeat);
        match (nearest, repeat) {
            (false, false) => &self.sampler_linear,
            (true, false) => &self.sampler_nearest_clamp,
            (false, true) => &self.sampler_linear_repeat,
            (true, true) => &self.sampler_nearest_repeat,
        }
    }

    fn make_default_tex_bg(&self) -> wgpu::BindGroup {
        self.build_tex_bg(&MaterialTextureSlots::default())
    }
}

/// Halton low-discrepancy sequence — the standard source of TAA sub-pixel jitter.
fn halton(mut index: u32, base: u32) -> f32 {
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
