//! Scene traversal and draw encoding.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use super::device::{depth_stencil_state, MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    class, msg0, msg1, msg2, msg3, msg4, msg6, sel, AutoreleasePool, Id, Owned, NIL,
};
use super::resources::{
    new_sampler, render_pipeline, upload_texture_as, white_texture, PipelineKey, SamplerKey,
};
use crate::cameras::Camera;
use crate::core::{BufferGeometry, Layers, ObjectKind};
use crate::lights::Light;
use crate::materials::{Material, MaterialKind};
use crate::math::{Matrix4, Vector3};
use crate::renderer::viewport::{ClearMode, PixelRect};
use crate::scene::Scene;
use crate::textures::Texture;

/// Views one frame can carry, matching `MAX_VIEWS` in `shaders.metal`.
/// A plain camera uses one, a headset two (the eyes), a CAD quad view four
/// (one per viewport — see [`crate::helpers::QuadView::metal_views`]).
pub const MAX_VIEWS: usize = 4;

/// Light counts, matching the array sizes in `shaders.metal`.
use crate::lights::shadow::{Caster, MAX_SHADOW_RECORDS};

mod frame;
mod collect;
mod resources;
mod pipelines;
mod shadows;
mod capture;

const MAX_DIR_LIGHTS: usize = 4;
/// Each face of a point light's shadow cube, texels — as the wgpu path.
const POINT_SHADOW_SIZE: u32 = 512;
const MAX_POINT_LIGHTS: usize = 8;
const MAX_SPOT_LIGHTS: usize = 4;
const MAX_HEMI_LIGHTS: usize = 2;

/// Buffer indices, matching the `[[buffer(n)]]` attributes in `shaders.metal`.
const VB_VERTICES: NSUInteger = 0;
const VB_FRAME: NSUInteger = 1;
const VB_DRAW: NSUInteger = 2;
const VB_INSTANCES: NSUInteger = 3;
const VB_SKIN: NSUInteger = 4;
const VB_BONES: NSUInteger = 5;
/// The morph-target buffer and its per-draw `MorphInfo` uniform.
const VB_MORPH_DATA: NSUInteger = 6;
const VB_MORPH: NSUInteger = 7;
/// The geometry's per-vertex extras: three `float4`s a vertex — `(uv1,
/// uv2)`, `(uv3, 0, 0)` and its `tangent` (xyz, handedness in w) — present
/// when it has any of them (`Draw::phys_flags.z`: 1 uv sets, 2 tangents).
const VB_UV_SETS: NSUInteger = 8;
/// Shell fur's sixteen `float4` parameter slots (`vs_fur`, and `fs_fur` at
/// `FB_FUR`).
const VB_FUR: NSUInteger = 9;
const FB_FUR: NSUInteger = 2;
const FB_FRAME: NSUInteger = 0;
const FB_DRAW: NSUInteger = 1;

/// How many frames of transient (per-frame) GPU buffers to keep alive.
///
/// `render` does not block on the GPU, so a buffer referenced by an in-flight
/// command buffer must outlive the call that made it. Three frames covers
/// triple buffering, which is the deepest a `CAMetalLayer` goes.
const FRAMES_IN_FLIGHT: usize = 3;

/// Frames a cached geometry or texture survives without being drawn.
///
/// Two seconds at 60 fps: long enough that an object leaving and re-entering
/// the view is not re-uploaded, short enough that a scene rebuilt every frame
/// does not accumulate. See [`MetalRenderer::set_cache_retention`].
pub const DEFAULT_CACHE_RETENTION: u64 = 120;

// --------------------------------------------------------------- GPU structs
//
// Twins of the structs in `shaders.metal`. Members are 16-byte aligned there,
// so they are here; `tests/metal_backend.rs` asserts the sizes.

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
    /// The colour attribute's alpha (1 without one): three.js multiplies a
    /// 4-component vertex colour into opacity.
    pub alpha: f32,
}

/// Set `THREERS_NO_SLIM_LINES=1` to upload lines as full `Vertex`es again.
///
/// An escape hatch for a change that alters how every line in a scene reaches
/// the GPU: the two paths are meant to be pixel-identical, and this is what
/// lets that be measured rather than asserted.
fn slim_lines_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("THREERS_NO_SLIM_LINES").is_none())
}

/// The slim vertex a line draw reads: 24 bytes against `Vertex`'s 48.
///
/// Lines are unlit and untextured, so normal, uv and pad are dead weight. At
/// connectome scale that halves the resident geometry — see `LineVertex` in
/// `shaders.metal`, whose layout this must match exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DirLightGpu {
    direction: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PointLightGpu {
    position: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SpotLightGpu {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HemiLightGpu {
    sky: [f32; 4],
    ground: [f32; 4],
    up: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct FrameUniforms {
    view_proj: [[f32; 16]; MAX_VIEWS],
    view: [[f32; 16]; MAX_VIEWS],
    camera_pos: [[f32; 4]; MAX_VIEWS],
    ambient: [f32; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    viewport: [f32; 4],
    counts: [u32; 4],
    /// `[view_count, 0, 0, 0]`.
    view_info: [u32; 4],
    dir: [DirLightGpu; MAX_DIR_LIGHTS],
    points: [PointLightGpu; MAX_POINT_LIGHTS],
    spots: [SpotLightGpu; MAX_SPOT_LIGHTS],
    hemis: [HemiLightGpu; MAX_HEMI_LIGHTS],
    /// `[rect-area light count, transmission capture mips, reflective
    /// transparency (0/1), the capture slice of view 0]` — w is the view's
    /// own index when a per-view pass has promoted it into slot 0.
    rect_counts: [u32; 4],
    rects: [RectLightGpu; MAX_RECT_LIGHTS],
    /// x: shadow records in use.
    shadow_info: [u32; 4],
    /// One per shadow-casting light, which names it in its own `color[3]`
    /// (record + 1) — `ShadowRecord` in `shaders.metal`.
    shadows: [ShadowRecordGpu; MAX_SHADOW_RECORDS],
}

/// three.js's transmission render target: the opaque scene drawn linear and
/// untone-mapped into `target`, then copied into `mipped` and its mips
/// generated, which the transmissive materials read.
struct MetalCapture {
    target: crate::metal::MetalRenderTarget,
    mipped: Owned,
    size: (u32, u32),
    levels: u32,
    /// One array slice per view, each the scene from that view.
    slices: u32,
}

/// One `RectAreaLight`: its centre, colour × intensity, and half width and
/// half height as vectors (three.js's `halfWidth`, `halfHeight`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RectLightGpu {
    position: [f32; 4],
    color: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

const MAX_RECT_LIGHTS: usize = 4;

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            view_proj: [Matrix4::identity().elements; MAX_VIEWS],
            view: [Matrix4::identity().elements; MAX_VIEWS],
            camera_pos: [[0.0; 4]; MAX_VIEWS],
            view_info: [1, 0, 0, 0],
            ambient: [0.0; 4],
            fog_color: [0.0; 4],
            fog_params: [0.0; 4],
            viewport: [1.0, 1.0, 1.0, 1.0],
            counts: [0; 4],
            dir: [DirLightGpu::default(); MAX_DIR_LIGHTS],
            points: [PointLightGpu::default(); MAX_POINT_LIGHTS],
            spots: [SpotLightGpu::default(); MAX_SPOT_LIGHTS],
            hemis: [HemiLightGpu::default(); MAX_HEMI_LIGHTS],
            rect_counts: [0; 4],
            rects: [RectLightGpu::default(); MAX_RECT_LIGHTS],
            shadow_info: [0; 4],
            shadows: [ShadowRecordGpu::default(); MAX_SHADOW_RECORDS],
        }
    }
}

impl FrameUniforms {
    /// Move view `index` into slot 0 and declare a single view.
    ///
    /// The non-layered shaders always read view 0, so a per-view pass hands
    /// them the view it is drawing.
    fn promote_view(&mut self, index: usize) {
        if index != 0 && index < MAX_VIEWS {
            self.view_proj[0] = self.view_proj[index];
            self.view[0] = self.view[index];
            self.camera_pos[0] = self.camera_pos[index];
        }
        self.view_info[0] = 1;
    }
}

/// An environment cube on the GPU: its faces (mipmapped), and its PMREM
/// CubeUV atlas with that atlas's lookup parameters (texel width and height,
/// top mip, 1) where it has one.
struct EnvGpu {
    cube: Owned,
    atlas: Option<Owned>,
    params: [f32; 4],
}

impl EnvGpu {
    fn upload(device: &MetalDevice, cube: &crate::textures::CubeTexture) -> Result<Self, MetalError> {
        let tex = crate::metal::resources::upload_cube(device, cube)?;
        let (atlas, params) = match &cube.cube_uv_atlas {
            Some(a) => (
                Some(crate::metal::resources::upload_cube_uv_atlas(
                    device,
                    a,
                    cube.format == crate::TextureFormat::Rgba8UnormSrgb,
                )?),
                [a.texel_width, a.texel_height, a.lod_max as f32, 1.0],
            ),
            None => (None, [0.0; 4]),
        };
        Ok(Self { cube: tex, atlas, params })
    }
}

/// Twin of `MorphInfo` in `shaders.metal`: the per-draw morph state.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MorphUniforms {
    /// Target count (0 = off), stride in float4s, flags (1 normal, 2 colour,
    /// 4 position), vertex count.
    info: [u32; 4],
    /// x: three.js's `morphTargetBaseInfluence`.
    base: [f32; 4],
    weights: [[f32; 4]; 16],
}

/// A geometry's morph targets on the GPU, laid out like three.js's morph
/// texture — see `apply_morph` in `shaders.metal`.
struct MorphBuffer {
    buffer: Owned,
    count: u32,
    stride: u32,
    flags: u32,
    vertex_count: u32,
}

/// What `vs_skinned` reads per vertex, alongside the ordinary `Vertex`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinVertex {
    joints: [f32; 4],
    weights: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct DrawUniforms {
    model: [f32; 16],
    normal_mat: [f32; 16],
    base_color: [f32; 4],
    emissive: [f32; 4],
    specular: [f32; 4],
    pbr: [f32; 4],
    misc: [f32; 4],
    flags: [u32; 4],
    /// Which of the extra texture slots are bound, as 0/1: emissive, normal,
    /// metallic-roughness, ambient occlusion. `flags.y` already carries the
    /// base-colour map and stays where it is so nothing else has to move.
    maps: [u32; 4],
    /// Scalars the maps modulate: normal-map scale, AO strength, then the
    /// environment's intensity and whether one is bound at all.
    map_params: [f32; 4],
    /// `[receives_shadow, is_sprite, 0, 0]`. Per-object rather than
    /// per-material, which is where `Object3D` keeps it.
    object: [u32; 4],
    /// Which variant-only maps are bound: alpha, light, bump, specular.
    extra: [u32; 4],
    /// `Material::map_uniforms()[0]`: `[lightMapIntensity, bumpScale, envMap
    /// bound, 0]`.
    map_scalars: [f32; 4],
    /// `Material::map_uniforms()[1]`: the `envMap`'s combine, reflectivity,
    /// refraction ratio and mapping.
    env_params: [f32; 4],
    /// `Material::map_uniforms()[2..5]`: its lookup rotation, flip folded in.
    env_dir: [[f32; 4]; 3],
    /// Each map's sampler as an index into the bound sampler pool, four to a
    /// vector in [`MAP_COUNT`] order — see `SAMPLER_POOL`.
    map_samplers: [[u32; 4]; 6],
    /// MeshPhysicalMaterial's scalars (zero for anything else):
    /// `[ior, specularIntensity, clearcoat, clearcoatRoughness]`,
    /// `[specularColor, sheen]`, `[sheenColor, sheenRoughness]`,
    /// `[iridescence, iridescenceIOR, iridescenceThicknessRange]`,
    /// `[anisotropy, anisotropyRotation, clearcoatNormalScale]`,
    /// `[transmission, thickness, attenuationDistance (0 = infinite),
    /// dispersion]`, `[attenuationColor, 0]`.
    physical: [[f32; 4]; 7],
    /// `[bitmask of the bound physical maps, specular antialiasing, 0, 0]`.
    phys_flags: [u32; 4],
    /// Each map's uv transform, two rows of three.js's `Texture.matrix`, in
    /// [`MAP_COUNT`] order: base, emissive, normal, metallic-roughness,
    /// ambient occlusion, alpha, light, bump, specular, matcap, then the
    /// physical maps ([`physical_maps`]).
    map_xform: [[f32; 4]; 2 * MAP_COUNT],
    /// The planet-rendering terms of Standard and Physical materials, as the
    /// wgpu backend's `params8`/`params10`–`params13`:
    /// `[emissive_night_side, sun angular radius, twilight wrap width, cloud
    /// scatter]`, `[twilight colour, scatter anisotropy]`, `[cloud shell
    /// height, cloud shadow strength, cloud longitude offset, cloud map
    /// bound]`, `[eclipse occluder centre, radius]`; and `[normal_scale.y,
    /// 0, 0, 0]`, the normal map's own y scale.
    planet: [[f32; 4]; 5],
}

/// How many material maps a draw can bind, `MAP_*` in `shaders.metal`.
const MAP_COUNT: usize = 22;
/// Where the physical maps start, in that order (`MAP_PHYS0`), and at which
/// texture slot they are bound.
const MAP_PHYS0: usize = 10;
const PHYS_TEXTURE0: usize = 16;
/// Sampler slots 0–9: one per distinct filter-and-wrap among a draw's maps.
const SAMPLER_POOL: usize = 10;
/// A `ShaderMaterial`'s `maps` — `u_tex0`..`u_tex3` — at fragment and
/// vertex textures 16–19: the physical maps' slots, which `fs_custom`
/// never reads. `u_samp`, linear and repeating, at sampler 15.
const USER_TEXTURE0: usize = 16;
const USER_SAMPLER_SLOT: usize = 15;
/// `ShaderMaterial::volumes`: `u_vol0` and `u_vol1` at textures 20 and 21,
/// `u_vol_samp` (linear, clamp-to-edge, as on wgpu) at sampler 9.
const USER_VOLUME0: usize = 20;
const USER_VOLUME_SAMPLER_SLOT: usize = 9;
const USER_VOLUME_SAMPLER: SamplerKey = SamplerKey {
    mag_linear: true,
    min_linear: true,
    mipmapped: false,
    mip_linear: false,
    wrap_s: 0,
    wrap_t: 0,
};
const USER_SAMPLER: SamplerKey = SamplerKey {
    mag_linear: true,
    min_linear: true,
    mipmapped: true,
    mip_linear: true,
    wrap_s: 1,
    wrap_t: 1,
};

/// MeshPhysicalMaterial's own maps, in `MAP_PHYS0` order, and whether each is
/// data (linear) rather than colour.
fn physical_maps(slots: &crate::materials::MaterialTextureSlots) -> [(Option<&Arc<Texture>>, bool); 12] {
    [
        (slots.specular_intensity_map.as_ref(), true),
        (slots.specular_color_map.as_ref(), false),
        (slots.clearcoat_map.as_ref(), true),
        (slots.clearcoat_roughness_map.as_ref(), true),
        (slots.clearcoat_normal_map.as_ref(), true),
        (slots.iridescence_map.as_ref(), true),
        (slots.iridescence_thickness_map.as_ref(), true),
        (slots.sheen_color_map.as_ref(), false),
        (slots.sheen_roughness_map.as_ref(), true),
        (slots.transmission_map.as_ref(), true),
        (slots.thickness_map.as_ref(), true),
        (slots.anisotropy_map.as_ref(), true),
    ]
}

/// The identity uv transform's two rows, for every map.
const IDENTITY_XFORMS: [[f32; 4]; 2 * MAP_COUNT] = {
    let mut x = [[0.0; 4]; 2 * MAP_COUNT];
    let mut i = 0;
    while i < MAP_COUNT {
        x[2 * i] = [1.0, 0.0, 0.0, 0.0];
        x[2 * i + 1] = [0.0, 1.0, 0.0, 0.0];
        i += 1;
    }
    x
};

/// [`DrawUniforms::map_xform`] for a material: each map's own transform, as
/// three.js gives every texture its own `matrix`.
fn map_xforms(slots: &crate::materials::MaterialTextureSlots) -> [[f32; 4]; 2 * MAP_COUNT] {
    let mut out = IDENTITY_XFORMS;
    let mut maps = [None; MAP_COUNT];
    maps[..10].copy_from_slice(&[
        slots.map.as_ref(),
        slots.emissive_map.as_ref(),
        slots.normal_map.as_ref(),
        slots.roughness_map.as_ref().or(slots.metalness_map.as_ref()),
        slots.ao_map.as_ref(),
        slots.alpha_map.as_ref(),
        slots.light_map.as_ref(),
        slots.bump_map.as_ref(),
        slots.specular_map.as_ref(),
        // The matcap is looked up by normal, not uv.
        None,
    ]);
    for (i, (t, _)) in physical_maps(slots).into_iter().enumerate() {
        maps[MAP_PHYS0 + i] = t;
    }
    for (i, t) in maps.into_iter().enumerate() {
        if let Some(t) = t {
            let [r0, r1] = t.uv_transform_rows();
            // Row 0's w: the uv set the map reads (`Texture.channel`).
            out[2 * i] = [r0[0], r0[1], r0[2], t.channel.min(3) as f32];
            out[2 * i + 1] = [r1[0], r1[1], r1[2], 0.0];
        }
    }
    out
}

impl Default for DrawUniforms {
    fn default() -> Self {
        Self {
            model: Matrix4::identity().elements,
            normal_mat: Matrix4::identity().elements,
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0; 4],
            specular: [0.0, 0.0, 0.0, 30.0],
            pbr: [1.0, 0.0, 0.0, 0.0],
            misc: [0.0, 1.0, 0.1, 1000.0],
            flags: [0; 4],
            maps: [0; 4],
            map_params: [1.0, 1.0, 1.0, 0.0],
            object: [0; 4],
            extra: [0; 4],
            map_scalars: [1.0, 1.0, 0.0, 0.0],
            env_params: [0.0, 1.0, 0.98, 0.0],
            env_dir: [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            map_samplers: [[0; 4]; 6],
            physical: [[0.0; 4]; 7],
            phys_flags: [0; 4],
            map_xform: IDENTITY_XFORMS,
            planet: [[0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [1.0, 0.0, 0.0, 0.0]],
        }
    }
}

// ------------------------------------------------------------------ pass I/O

/// The attachments of one render pass.
///
/// Produced by [`MetalRenderTarget::attachments`](super::MetalRenderTarget::attachments)
/// for offscreen work and by [`SurfaceFrame`](super::SurfaceFrame) for a window;
/// [`crate::metal::renderer::PassAttachments::from_raw`] covers integrating with an app that already owns
/// its own Metal textures.
#[derive(Clone, Copy)]
pub struct PassAttachments {
    pub(crate) color: Id,
    pub(crate) resolve: Id,
    pub(crate) depth: Id,
    /// Where a multisampled `depth` is resolved when the pass stores depth;
    /// nil for none.
    pub(crate) depth_resolve: Id,
    pub(crate) drawable: Id,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) color_format: NSUInteger,
    pub(crate) depth_format: NSUInteger,
    pub(crate) sample_count: u32,
    pub(crate) reverse_z: bool,
}

impl PassAttachments {
    /// Wrap textures this crate did not create.
    ///
    /// # Safety
    /// `color`, `resolve`, `depth` and `drawable` must be `nil` or live objects
    /// of the corresponding Metal type, valid until the next `render` returns.
    /// `color_format` and `sample_count` must describe `color`, or pipeline
    /// creation fails at draw time.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_raw(
        color: Id,
        resolve: Id,
        depth: Id,
        drawable: Id,
        width: u32,
        height: u32,
        color_format: NSUInteger,
        depth_format: NSUInteger,
        sample_count: u32,
    ) -> Self {
        Self {
            color,
            resolve,
            depth,
            depth_resolve: NIL,
            drawable,
            width,
            height,
            color_format,
            depth_format,
            sample_count,
            reverse_z: false,
        }
    }

    /// Resolve a multisampled depth into `texture` — single-sampled,
    /// `DEPTH32_FLOAT`, the attachments' size and slices — whenever the pass
    /// stores depth (`MetalRenderer::set_store_depth`), so it can be sampled
    /// as a `depth2d` afterwards.
    ///
    /// # Safety
    /// `texture` must be nil or a live `MTLTexture` as described, valid until
    /// the next `render` returns.
    pub unsafe fn with_depth_resolve(mut self, texture: Id) -> Self {
        self.depth_resolve = texture;
        self
    }

    /// Invert the depth convention: near at 1, far at 0, and a `greater` depth
    /// test. The depth clear value follows.
    ///
    /// Required on visionOS — the compositor accepts nothing else (`drawable.h`:
    /// "It only supports reverse-Z depth"). Worth having anyway: floating-point
    /// depth has most of its precision near 0, and reverse-Z puts that where the
    /// far plane is, which is where z-fighting otherwise shows up.
    ///
    /// The projection matrix has to agree — `RenderView::projection_matrix` must
    /// map the near plane to 1. A compositor-supplied projection already does.
    pub fn with_reverse_z(mut self, reverse: bool) -> Self {
        self.reverse_z = reverse;
        self
    }

    /// Attachment size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether depth runs near-at-1.
    pub fn reverse_z(&self) -> bool {
        self.reverse_z
    }
}

/// One view of a frame: a camera, and where its pixels land.
///
/// A window renders one of these. A headset renders two — the eyes — which is
/// what [`crate::metal::renderer::MetalRenderer::render_views`] takes and what the visionOS compositor
/// hands over per frame.
#[derive(Clone, Debug)]
pub struct RenderView {
    /// World → view.
    pub view_matrix: Matrix4,
    /// View → clip. Depth must land in `[0, 1]`; reverse-Z (near at 1) is fine
    /// as long as the pass is marked
    /// [`with_reverse_z`](PassAttachments::with_reverse_z).
    pub projection_matrix: Matrix4,
    /// Eye position in world space, for specular and fog.
    pub position: Vector3,
    /// Layer mask; an object is drawn when it intersects this.
    pub layers: Layers,
    /// Array slice of the colour and depth attachments to render into.
    pub slice: u32,
    /// Sub-rectangle of the attachment, or `None` for all of it.
    pub viewport: Option<MTLViewport>,
    /// Colour texture for this view alone, or `nil` to use the pass's. Set for
    /// a compositor layout that gives each eye its own texture.
    pub color: Id,
    /// Depth texture for this view alone, or `nil` to use the pass's.
    pub depth: Id,
}

impl RenderView {
    /// A view from any [`Camera`], filling the whole attachment.
    pub fn from_camera(camera: &dyn Camera) -> Self {
        Self {
            view_matrix: camera.view_matrix(),
            projection_matrix: camera.projection_matrix(),
            position: camera.position(),
            layers: camera.layers(),
            slice: 0,
            viewport: None,
            color: NIL,
            depth: NIL,
        }
    }

    /// A view from explicit matrices — what an XR compositor provides.
    pub fn from_matrices(view_matrix: Matrix4, projection_matrix: Matrix4) -> Self {
        let inverse = view_matrix.invert().elements;
        Self {
            view_matrix,
            projection_matrix,
            position: Vector3::new(inverse[12], inverse[13], inverse[14]),
            layers: Layers::default(),
            slice: 0,
            viewport: None,
            color: NIL,
            depth: NIL,
        }
    }

    /// Render into array slice `slice` of the attachment.
    pub fn with_slice(mut self, slice: u32) -> Self {
        self.slice = slice;
        self
    }

    /// Render into a sub-rectangle of the attachment.
    pub fn with_viewport(mut self, viewport: MTLViewport) -> Self {
        self.viewport = Some(viewport);
        self
    }

    /// Render into textures of this view's own, rather than the pass's.
    ///
    /// # Safety
    /// Both must be `nil` or live `MTLTexture`s matching the pass's format and
    /// sample count, valid until `render_views` returns.
    pub unsafe fn with_textures(mut self, color: Id, depth: Id) -> Self {
        self.color = color;
        self.depth = depth;
        self
    }
}

/// What one [`MetalRenderer::render`] call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalRenderStats {
    /// Draw calls encoded.
    pub draw_calls: u32,
    /// Triangles submitted, counting instances.
    pub triangles: u32,
    /// Objects skipped because their geometry had no `position` attribute.
    pub skipped: u32,
    /// Items outside a view's frustum, not encoded (counted per view).
    pub culled: u32,
    /// Geometries uploaded this frame (a cache miss or a version bump).
    pub geometry_uploads: u32,
    /// Textures uploaded this frame.
    pub texture_uploads: u32,
    /// Bytes currently held in cached vertex + index buffers. The figure that
    /// decides whether a large scene fits in memory or pages, and the one the
    /// slim line layout exists to halve.
    pub geometry_bytes: u64,
}

// -------------------------------------------------------------------- caches

struct CachedGeometry {
    /// Held to pin the allocation: the cache key is the `Arc`'s address, and a
    /// dropped geometry's address can otherwise be handed to a new one.
    _geometry: Arc<BufferGeometry>,
    version: u32,
    vertices: Owned,
    /// Joint indices and weights, built only when the geometry carries them.
    /// A parallel buffer rather than a wider `Vertex`: skinned meshes are a
    /// minority and 32 bytes a vertex on every other mesh is not a trade worth
    /// making.
    skin: Option<Owned>,
    indices: Option<Owned>,
    index_count: usize,
    vertex_count: usize,
    /// Line-list indices for `wireframe: true` materials, from
    /// [`BufferGeometry::wireframe_indices`]: polygon outlines when the
    /// geometry knows its polygons, so a quad draws without its diagonal.
    /// Built on first wireframe use rather than with the rest, since most
    /// meshes are never drawn that way; a version bump replaces the whole
    /// entry, which drops this with it.
    wire: Option<(Owned, usize)>,
    /// Morph targets, when the geometry has any. Rebuilt with the rest on a
    /// version bump.
    morph: Option<MorphBuffer>,
    /// The `uv1`/`uv2`/`uv3` sets and tangents ([`VB_UV_SETS`]), when it
    /// has any, and which (bit 1 uv sets, bit 2 tangents).
    uv_sets: Option<(Owned, u32)>,
    /// Size of the GPU buffers this entry holds, for `geometry_bytes`.
    bytes: u64,
    /// Object-space bounds, for frustum culling (`item_sphere`); `None` with
    /// no positions, which is never culled.
    sphere: Option<crate::math::Sphere>,
    /// Which layout `vertices` holds. A geometry drawn as lines and then as a
    /// mesh (or the reverse) has to be re-uploaded, not reinterpreted.
    slim: bool,
    last_used: u64,
}

struct CachedTexture {
    _texture: Arc<Texture>,
    texture: Owned,
    sampler_key: SamplerKey,
    last_used: u64,
    /// For a streamed texture, the frame count last written into it.
    stream_frames: u64,
}

/// One item collected from the scene graph, ready to encode.
/// A shadow-casting light's shadow — `ShadowRecord` in `shaders.metal`, and
/// the wgpu renderer's `ShadowRecordGpu`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ShadowRecordGpu {
    info: [u32; 4],
    params: [f32; 4],
    extra: [f32; 4],
    pcss: [f32; 4],
    vp: [[f32; 16]; 4],
}

/// What one shadow pass draws with — `ShadowPass` in `shaders.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ShadowPassGpu {
    shadow_vp: [f32; 16],
    point_shadow_pos: [f32; 4],
}

/// The OIT accumulation's targets: the resolved `accum` and `reveal` the
/// composite reads, and their multisampled twins under MSAA.
struct OitTargets {
    size: (u32, u32, u32),
    accum: Owned,
    reveal: Owned,
    accum_ms: Option<Owned>,
    reveal_ms: Option<Owned>,
}

/// Every shadow-casting light this frame, in the order its record takes.
type Casters = Vec<Caster>;

struct DrawItem {
    /// Shell fur's parameters (`FurMaterial::data`) when the material is fur.
    fur: Option<[f32; 64]>,
    /// An `fx` surface's uniforms and arrays (`crate::fx::surface::metal_bytes`)
    /// when the material is one.
    fx_surface: Option<Box<[f32; crate::fx::surface::METAL_FLOATS]>>,
    /// A `ShaderMaterial` of the user's — not fur or an `fx` surface, which
    /// have ports of their own — drawn through its own fragment (`msl`, or
    /// its WGSL translated; see `MetalRenderer::custom_shader`).
    custom: bool,
    geometry: Arc<BufferGeometry>,
    material: Arc<Material>,
    world: Matrix4,
    layers: Layers,
    primitive: NSUInteger,
    /// Per-instance matrices, until they are uploaded — then the index of the
    /// buffer holding them, so a skipped item cannot shift another item's
    /// buffer out from under it.
    instances: Option<Vec<f32>>,
    instance_slot: Option<usize>,
    instance_count: usize,
    /// An `InstancedMesh`'s morph texture (three.js `morphTexture`): one row
    /// of `row_len` floats per instance, until uploaded after the matrices in
    /// the instance buffer — then `(row_len, offset in floats)`, which is
    /// where `apply_morph` reads them.
    instance_morph: Option<(Vec<f32>, usize)>,
    instance_morph_layout: Option<(u32, u32)>,
    /// `InstancedMesh.colors` (three.js `instanceColor`), four floats an
    /// instance, until uploaded after the matrices and morph rows — then
    /// where they start in the instance buffer, in float4s (0: none).
    instance_colors: Option<Vec<f32>>,
    /// `InstancedMesh::custom`, two float4 a instance: what a custom vertex
    /// stage reads as `icustom0`/`icustom1`.
    instance_custom: Option<Vec<f32>>,
    instance_color_offset: u32,
    view_depth: f32,
    render_order: i32,
    transparent: bool,
    /// three.js's transmissive material (`Material::transmissive`): drawn
    /// after the opaque items, reading them from the transmission capture.
    transmissive: bool,
    /// Glass blended over what is behind it, tinting it by channel
    /// (`TransparencyMode::Glass` and `Oit`; `phys_flags.w` in the shader).
    blended_glass: bool,
    /// Order-independent transparency (`TransparencyMode::Oit`).
    oit: bool,
    /// A `ShaderMaterial` with `screen_space`: it reads the opaque scene
    /// behind it (`ss_color_tex` / `ss_depth_tex`) from the capture, and is
    /// left out of it, as the wgpu screen-space pass leaves it out.
    screen_space: bool,
    /// Left out of the capture pass running now (see `capture_transmission`).
    capture_skip: bool,
    /// The side the item is drawn with in place of its material's — the
    /// capture's back-face pass draws double-sided glass as BackSide.
    side_override: Option<u32>,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    state: crate::materials::RenderState,
    /// A skinned mesh's bone matrices, until they are uploaded — then the
    /// index of the transient buffer holding them, for the same reason
    /// `instances` works that way.
    bones: Option<Vec<f32>>,
    bone_slot: Option<usize>,
    /// Whether this object draws into the shadow map. Off by default, as it is
    /// on [`Object3D`] — a scene where everything casts is rarely what anyone
    /// means, and it costs a full depth pass over geometry that never shows.
    cast_shadow: bool,
    /// Whether this object is darkened by the shadow map. Also per-object and
    /// also off by default, and for a sharper reason than casting: a surface
    /// that receives without anything casting is unchanged, but one that casts
    /// without receiving shadows itself through acne.
    receive_shadow: bool,
    /// The span of the index (vertices, unindexed) this item draws —
    /// three.js's draw range cut to the item's group. Always set; a full span
    /// is `0..draw_count`.
    range: (u32, u32),
    /// The object's `morph_influences` (empty when it has none).
    morph_influences: Vec<f32>,
    /// A wireframe drawn over a partial span: its line list, in this frame's
    /// transient buffers, and the index count.
    wire_span: Option<(usize, usize)>,
    /// For a `Sprite`: three.js's `sprite_vert` parameters — centre minus
    /// one half (x, y), rotation, and 1 when the quad scales with view depth
    /// (size attenuation off under a perspective view). `vs_mesh` billboards
    /// the draw's unit quad with them; see `sprite_vertex` in the shader.
    sprite: Option<[f32; 4]>,
}

// ------------------------------------------------------------------ renderer

/// Draws a [`Scene`] with Metal.
///
/// Owns the caches (geometry, textures, samplers, pipeline states) and nothing
/// about *where* the pixels go — pass [`PassAttachments`] for that, so one
/// renderer can serve a window and an offscreen target at once.
///
/// # Coverage
///
/// Meshes, instanced meshes, line segments and points, with `Basic`, `Lambert`,
/// `Phong`, `Standard`, `Physical`, `Normal`, `Depth`, `Toon`, `Matcap`, `Line`,
/// `Points` and `Sprite` materials, one base-colour map, and directional, point,
/// spot, hemisphere and ambient lights. Materials outside that set draw unlit in
/// their base colour rather than failing.
///
/// Environment/IBL and the normal, roughness, metalness, AO and emissive map
/// slots are implemented here, and `tests/backend_parity.rs` holds them to the
/// wgpu renderer: emissive maps agree to 0.0024 rms over a luma grid and
/// environment reflections to 0.0009 at mirror roughness.
///
/// Skinning is here too: a `SkinnedMesh` blends through its skeleton's bone
/// matrices in `vs_skinned`, four influences a vertex, and measures identical
/// to the wgpu path in the same pose.
///
/// Shadows are here from every casting light, as on the wgpu path: a
/// depth-only pass per light into a layer of its kind's map array, filtered
/// over the same twelve-tap disc with normal bias, and reused while nothing
/// they show changes (`ShadowMap`). `cast_shadow` and `receive_shadow` are
/// read per object, as the wgpu path reads them, and `tests/shadow_fixtures.rs`
/// and `tests/multi_light_shadows.rs` hold the two to the same placement.
///
/// A point light's map is six layers, its cube faces, written by the only
/// shadow pass here with a fragment stage, because they store *radial*
/// distance rather than projected depth, and looked up by direction with the
/// face chosen by hand. A point light and a spot light in the same place
/// throw the same shadow, 0.0785 of darkening over eight tiles against 0.0786
/// over eight.
///
/// Instanced meshes with non-uniform per-instance scale shade with skewed
/// normals.
///
/// Morph targets are here: `morphAttributes` position, normal and colour,
/// blended in the vertex stage from one float4 buffer per geometry laid out
/// like three.js's morph texture (see `apply_morph` in the shader), before
/// skinning, in every mesh pass including the shadow ones. Groups with a
/// material array draw one item per group, and the draw range cuts every
/// draw — three.js's render list, item for item.
///
/// The direct-light BRDF matches too, as of the `1/pi` diffuse normalisation
/// and height-correlated Smith visibility: every map slot measures within
/// 0.016 rms of the wgpu path. What is still missing there is the roughness
/// floor and the geometric specular antialiasing it applies.
pub struct MetalRenderer {
    device: MetalDevice,
    pipelines: HashMap<PipelineKey, Owned>,
    samplers: HashMap<SamplerKey, Owned>,
    geometries: HashMap<usize, CachedGeometry>,
    textures: HashMap<usize, CachedTexture>,
    /// Writes streamed (video) frames into their textures.
    streams: super::video::MetalStreams,
    /// Depth-stencil states by (`MTLCompareFunction`, write), made on first
    /// use: every material picks its own from `depthTest` / `depthWrite` /
    /// `depthFunc`.
    depth_states: HashMap<(NSUInteger, bool), Owned>,
    white: Owned,
    /// See [`zero_volume`](crate::metal::resources::zero_volume).
    zero_volume: Owned,
    /// three.js's `RectAreaLight` tables, at fragment texture 28.
    ltc: Owned,
    /// three.js's transmission render target — see [`MetalCapture`].
    transmission: Option<MetalCapture>,
    /// The capture the colour pass binds at fragment texture 29
    /// (`capture_dummy` when `None`), and its trilinear sampler at 14.
    capture_bound: Option<Id>,
    /// A 1×1 black array in the capture's shape, for frames with nothing
    /// transmissive.
    capture_dummy: Owned,
    /// A 1×1 depth texture, bound at fragment texture 30 when there is no
    /// capture depth to read.
    capture_depth_dummy: Owned,
    transmission_sampler: Owned,
    /// Bound at the cube slot when a scene has no environment.
    black_cube: Owned,
    /// The shadow maps: a depth array per kind, a layer per casting light
    /// (six, its cube faces, per point light), with the directional and spot
    /// arrays' sizes and every array's layer count. Allocated on the first
    /// frame a light asks to cast, because the size comes from the light and
    /// no light exists when the renderer is built.
    shadow_map: Option<Owned>,
    shadow_size: u32,
    spot_shadow_map: Option<Owned>,
    spot_shadow_size: u32,
    point_shadow_map: Option<Owned>,
    shadow_layers: [usize; 3],
    /// What the maps were last drawn from (see `shadow_fingerprint`).
    /// Each shadow pass's content hash when last drawn, so an unchanged pass
    /// is skipped.
    shadow_pass_hashes: Vec<Option<u64>>,
    /// Depth-only pipeline and comparison sampler for that map.
    shadow_pipeline: Option<Owned>,
    /// The point variant, which writes radial distance from a fragment stage.
    point_shadow_pipeline: Option<Owned>,
    /// Directional/spot shadows for casters cut by `alphaTest` and a map.
    cut_shadow_pipeline: Option<Owned>,
    shadow_sampler: Option<Owned>,
    /// Bound at the shadow slot when nothing casts. Metal requires every
    /// resource a function declares to be bound, and a 1x1 depth texture is
    /// cheaper than a second fragment function.
    empty_shadow: Owned,
    /// The uploaded environment, keyed by the `CubeTexture` it came from so a
    /// scene that keeps the same environment across frames uploads it once.
    /// Environments are large — six faces and a mip chain — and rebuilding one
    /// per frame would cost more than everything else this backend does.
    env_cube: Option<(crate::renderer::gpu_texture::CubeCacheKey, EnvGpu)>,
    /// Materials' own `envMap`s, keyed by the cube's upload
    /// (`cube_cache_key`: its stable id, format and size).
    material_envs: HashMap<crate::renderer::gpu_texture::CubeCacheKey, EnvGpu>,
    /// three.js `WebGLCubeUVMaps`: each cube texture lighting a
    /// Standard/Physical material, prefiltered through PMREM once, by its
    /// source's key — as the wgpu renderer does.
    pmrem_cache: HashMap<crate::renderer::gpu_texture::CubeCacheKey, Arc<crate::textures::CubeTexture>>,
    /// The scene environment's intensity and lookup rows, for the draws that
    /// read it.
    env_light: (f32, [[f32; 4]; 3]),
    identity_instance: Owned,
    transients: Vec<Vec<Owned>>,
    frame: u64,
    cache_retention: u64,
    /// Keep the depth attachment after the pass, for readers like SSAO.
    store_depth: bool,
    /// Keep the frame's attachments for a later pass that loads them —
    /// marks drawn over it: a multisampled colour survives its resolve, and
    /// depth is stored (multisampled as well as resolved).
    keep_attachments: bool,
    /// The target the caller asked for is plain `Unorm`, drawn here through
    /// an sRGB one (`set_encoded_output`).
    encoded_output: bool,
    /// This frame has OIT items and draws them in their own passes; and which
    /// of `encode_pass`'s two jobs is running — 0 everything else, 1 the OIT
    /// accumulation (`encode_oit`).
    oit_active: bool,
    oit_phase: u8,
    oit_targets: Option<OitTargets>,
    /// The composite's two halves, by colour format and sample count.
    oit_composite: HashMap<(NSUInteger, u32), (Owned, Owned)>,
    /// Line width in pixels; 0 keeps one-pixel hardware lines.
    line_width: f32,
    /// three.js `renderer.toneMapping` / `toneMappingExposure`, applied in
    /// the fragment stage to every material that is `tone_mapped`.
    tone_mapping: (crate::ToneMapping, f32),
    /// See [`set_specular_antialiasing`](MetalRenderer::set_specular_antialiasing).
    specular_antialiasing: bool,
    /// See [`set_reflective_transparency`](MetalRenderer::set_reflective_transparency).
    reflective_transparency: bool,
    /// three.js `renderer.shadowMap.type`; a light's `ShadowSettings::filter`
    /// overrides it.
    shadow_type: crate::lights::ShadowType,
    /// Last `geometry_bytes` reported, so the log fires on change not per frame.
    /// Atomic because the frame build runs behind `&self`.
    logged_geometry_bytes: std::sync::atomic::AtomicU64,
    /// three.js viewport / scissor / auto-clear state. See
    /// [`crate::metal::viewport`].
    pub(super) viewport_state: super::viewport::MetalViewportState,
    /// three.js `shadowMap.autoUpdate` / `needsUpdate`.
    shadow_flags: crate::renderer::ShadowMap,
    /// Caster kinds and directional map size the maps were last drawn for.
    shadow_sig: Option<(usize, usize, usize, u32, u32)>,
    shadow_updates: u64,
    /// The 2D overlay's texture and blend pipelines (`draw_overlay`).
    overlay: super::overlay::MetalOverlay,
    /// Each custom shader's library — the built-in source with its MSL
    /// appended — by [`custom_shader_id`]; `None` when it did not compile,
    /// which has been said once and draws the fallback colour from then on.
    custom_libs: HashMap<u64, Option<Owned>>,
    /// Each WGSL fragment's MSL by the hash of the WGSL, translated once;
    /// `None` when it does not translate (said once, with the reason).
    translated: HashMap<u64, Option<Arc<str>>>,
}

impl MetalRenderer {
    /// Acquire the default device and build the renderer.
    pub fn new() -> Result<Self, MetalError> {
        Self::with_device(MetalDevice::new()?)
    }

    /// Build on an existing device — use this to share one device (and one
    /// compiled shader library) between a window and an offscreen target.
    pub fn with_device(device: MetalDevice) -> Result<Self, MetalError> {
        let mut depth_states = HashMap::new();
        // The shadow passes' state, forward-Z with writes.
        depth_states.insert(
            (compare::LESS, true),
            depth_stencil_state(&device, compare::LESS, true)?,
        );
        let white = white_texture(&device)?;
        let zero_volume = crate::metal::resources::zero_volume(&device)?;
        let ltc = crate::metal::resources::upload_ltc(&device)?;
        let capture_dummy = crate::metal::resources::black_texture_array(&device)?;
        let capture_depth_dummy = crate::metal::resources::new_texture_2d(
            &device,
            1,
            1,
            pixel_format::DEPTH32_FLOAT,
            texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        let transmission_sampler = new_sampler(
            &device,
            SamplerKey { mag_linear: true, min_linear: true, mipmapped: true, mip_linear: true, wrap_s: 0, wrap_t: 0 },
        )?;
        let black_cube = crate::metal::resources::black_cube(&device)?;
        let empty_shadow = crate::metal::resources::empty_shadow_map(&device)?;
        let identity_instance =
            device.new_buffer(bytemuck::cast_slice(&[Matrix4::identity().elements]))?;
        let viewport_state = super::viewport::MetalViewportState::new(device.clone());
        // The samplers every draw falls back to, made up front: a draw with no
        // texture would otherwise find neither in the cache and bind nil in
        // every sampler slot, which Metal's validation layer rejects.
        let mut samplers = HashMap::new();
        for key in [SamplerKey::default_linear(), SamplerKey::environment()] {
            samplers.insert(key, new_sampler(&device, key)?);
        }
        Ok(Self {
            device,
            pipelines: HashMap::new(),
            samplers,
            geometries: HashMap::new(),
            textures: HashMap::new(),
            streams: Default::default(),
            depth_states,
            white,
            zero_volume,
            ltc,
            capture_dummy,
            capture_depth_dummy,
            transmission: None,
            capture_bound: None,
            transmission_sampler,
            black_cube,
            material_envs: HashMap::new(),
            pmrem_cache: HashMap::new(),
            env_light: (1.0, crate::materials::environment_lookup_rows(Default::default(), true)),
            env_cube: None,
            shadow_map: None,
            shadow_size: 0,
            spot_shadow_map: None,
            spot_shadow_size: 0,
            point_shadow_map: None,
            shadow_layers: [0; 3],
            shadow_pass_hashes: Vec::new(),
            shadow_pipeline: None,
            point_shadow_pipeline: None,
            cut_shadow_pipeline: None,
            shadow_sampler: None,
            empty_shadow,
            identity_instance,
            transients: (0..FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect(),
            frame: 0,
            cache_retention: DEFAULT_CACHE_RETENTION,
            store_depth: false,
            keep_attachments: false,
            encoded_output: false,
            oit_active: false,
            oit_phase: 0,
            oit_targets: None,
            oit_composite: HashMap::new(),
            line_width: 0.0,
            tone_mapping: (crate::ToneMapping::None, 1.0),
            specular_antialiasing: false,
            reflective_transparency: false,
            shadow_type: crate::lights::ShadowType::default(),
            logged_geometry_bytes: std::sync::atomic::AtomicU64::new(0),
            viewport_state,
            shadow_flags: Default::default(),
            shadow_sig: None,
            shadow_updates: 0,
            overlay: Default::default(),
            custom_libs: HashMap::new(),
            translated: HashMap::new(),
        })
    }

    /// three.js `renderer.shadowMap`, as on the wgpu
    /// [`Renderer::shadow_map`](crate::Renderer::shadow_map): with
    /// `auto_update` off, a render redraws the shadow maps only when
    /// `needs_update` is set (and clears it). See
    /// [`ShadowMap`](crate::renderer::ShadowMap).
    pub fn shadow_map(&self) -> &crate::renderer::ShadowMap {
        &self.shadow_flags
    }

    pub fn shadow_map_mut(&mut self) -> &mut crate::renderer::ShadowMap {
        &mut self.shadow_flags
    }

    /// Blend an [`Overlay`](crate::overlay::Overlay) over the frame in `pass`,
    /// uploading only the rows that changed — the Metal side of
    /// [`Renderer::draw_overlay`](crate::renderer::Renderer::draw_overlay).
    ///
    /// [`render`](Self::render) presents a pass's drawable, which would put the
    /// frame on screen before the overlay; so render with
    /// [`without_drawable`](PassAttachments::without_drawable) and hand this
    /// the full attachments, and it presents after blending.
    pub fn draw_overlay(
        &mut self,
        overlay: &mut crate::overlay::Overlay,
        pass: &PassAttachments,
    ) -> Result<(), MetalError> {
        let (width, height) = overlay.size();
        // An empty band still composites: the frame underneath is new.
        let bands = overlay.dirty_bands();
        self.overlay
            .composite(&self.device, overlay.pixels(), width, height, Some(&bands), pass)?;
        overlay.mark_uploaded();
        Ok(())
    }

    /// As [`draw_overlay`](Self::draw_overlay), from straight-alpha sRGB RGBA8
    /// bytes the size of the frame, uploaded whole.
    pub fn draw_overlay_rgba8(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        pass: &PassAttachments,
    ) -> Result<(), MetalError> {
        self.overlay.composite(&self.device, rgba, width, height, None, pass)
    }

    /// How many renders so far drew the shadow maps.
    pub fn shadow_map_updates(&self) -> u64 {
        self.shadow_updates
    }

    /// The device this renderer draws with.
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Drop every cached GPU resource. The next frame re-uploads what it needs.
    pub fn clear_caches(&mut self) {
        self.geometries.clear();
        self.textures.clear();
    }

    /// Geometries and textures currently cached on the GPU.
    pub fn cache_sizes(&self) -> (usize, usize) {
        (self.geometries.len(), self.textures.len())
    }

    /// How many frames an unused geometry or texture stays on the GPU before it
    /// is dropped. Default [`DEFAULT_CACHE_RETENTION`].
    ///
    /// The cache holds an `Arc` to everything it has uploaded, so without an
    /// upper bound a scene rebuilt each frame would pin every geometry it has
    /// ever drawn — in GPU memory *and* in system memory. Raise it if objects
    /// come and go over a longer cycle than the default; `u64::MAX` never
    /// evicts.
    /// Draw lines as coverage-weighted screen-space quads of this pixel width.
    /// 0 (the default) keeps hardware lines, which are all-or-nothing at one
    /// pixel and so flicker for anything thinner.
    pub fn set_line_width(&mut self, px: f32) {
        self.line_width = px.max(0.0);
    }

    /// three.js `renderer.toneMapping` and `toneMappingExposure`, the same
    /// curves as the wgpu backend's [`crate::Renderer::set_tone_mapping`]:
    /// applied per material, before the sRGB encode, to every material whose
    /// `tone_mapped` is set (normal and depth materials never are).
    pub fn set_tone_mapping(&mut self, mode: crate::ToneMapping, exposure: f32) {
        self.tone_mapping = (mode, exposure);
    }

    /// The tone mapping set by [`set_tone_mapping`](Self::set_tone_mapping).
    pub fn tone_mapping(&self) -> (crate::ToneMapping, f32) {
        self.tone_mapping
    }

    /// Geometric specular antialiasing on Standard and Physical materials in
    /// place of three.js's `geometryRoughness`, as
    /// [`crate::Renderer::set_specular_antialiasing`]. Off by default.
    pub fn set_specular_antialiasing(&mut self, on: bool) {
        self.specular_antialiasing = on;
    }

    pub fn specular_antialiasing(&self) -> bool {
        self.specular_antialiasing
    }

    /// threers's reflective transparency on Standard and Physical, as
    /// [`crate::Renderer::set_reflective_transparency`]: a transparent
    /// surface's alpha raised by its specular reflectance. Off by default —
    /// three.js blends the whole colour by `opacity`.
    /// How shadow maps are filtered — three.js's `renderer.shadowMap.type`,
    /// as `Renderer::set_shadow_type`.
    pub fn set_shadow_type(&mut self, kind: crate::lights::ShadowType) {
        self.shadow_type = kind;
    }

    pub fn shadow_type(&self) -> crate::lights::ShadowType {
        self.shadow_type
    }

    pub fn set_reflective_transparency(&mut self, on: bool) {
        self.reflective_transparency = on;
    }

    pub fn reflective_transparency(&self) -> bool {
        self.reflective_transparency
    }

    /// Keep the depth attachment after each pass so a later pass can sample it.
    /// Off by default: it costs bandwidth that nothing otherwise reads.
    pub fn set_store_depth(&mut self, on: bool) {
        self.store_depth = on;
    }

    /// Keep every attachment of the frame for a pass that draws over it
    /// afterwards — the colour multisampled as well as resolved, and the
    /// depth. Off by default, like [`set_store_depth`](Self::set_store_depth).
    pub fn set_keep_attachments(&mut self, on: bool) {
        self.keep_attachments = on;
    }

    /// Stand in for a plain `Unorm` target while drawing into an sRGB one,
    /// as wgpu's renderer draws into `Rgba8Unorm`: a WGSL `ShaderMaterial`'s
    /// `framebuffer_encode` encodes, and what it returns is stored as it
    /// stands rather than encoded again. The built-in materials look the same
    /// either way. Off by default.
    pub fn set_encoded_output(&mut self, on: bool) {
        self.encoded_output = on;
    }

    pub fn set_cache_retention(&mut self, frames: u64) {
        self.cache_retention = frames;
    }

    // ------------------------------------------------------------ collection

    // --------------------------------------------------------------- uploads

    // -------------------------------------------------------------- encoding

}

/// The clear colour for `color` — every colour clear goes through here: the
/// pass's own (`loadAction` clear), the scissored fill a partial clear draws,
/// and [`MetalRenderer::clear`]. [`Color`](crate::math::Color) is linear, and
/// so is `MTLClearColor`; an sRGB attachment encodes on write, for the clear
/// and for the fill's fragment output alike.
pub(crate) fn clear_color(color: crate::math::Color, alpha: f32) -> MTLClearColor {
    MTLClearColor {
        red: color.r as f64,
        green: color.g as f64,
        blue: color.b as f64,
        alpha: alpha as f64,
    }
}

/// Whether these views can share one layered pass: more than one of them, all
/// into the pass's own texture, same viewport, distinct array slices, and no
/// MSAA resolve (a pass resolves one slice, so two eyes need two passes).
fn layered(views: &[RenderView], pass: &PassAttachments) -> bool {
    if views.len() < 2 || !pass.resolve.is_null() {
        return false;
    }
    let first = &views[0];
    let same_viewport = |a: &Option<MTLViewport>, b: &Option<MTLViewport>| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.origin_x == b.origin_x
                && a.origin_y == b.origin_y
                && a.width == b.width
                && a.height == b.height
        }
        _ => false,
    };
    views.iter().all(|v| {
        v.color.is_null() && v.depth.is_null() && same_viewport(&v.viewport, &first.viewport)
    }) && views
        .iter()
        .enumerate()
        .all(|(i, v)| views[..i].iter().all(|w| w.slice != v.slice))
}

// ------------------------------------------------------------------- helpers

/// `setVertexBytes:length:atIndex:` / `setFragmentBytes:length:atIndex:`.
///
/// Metal copies the bytes into the command buffer, so `T` needs to live only
/// for the call. Capped at 4 KB by the API — both uniform structs are far under.
/// The id a custom shader's library and pipelines are cached under: a hash of
/// its MSL, never 0 (which means "none").
fn custom_shader_id(msl: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    msl.hash(&mut h);
    h.finish().max(1)
}

unsafe fn set_bytes<T>(encoder: Id, vertex_stage: bool, value: &T, index: NSUInteger) {
    debug_assert!(std::mem::size_of::<T>() <= 4096);
    let sel = if vertex_stage {
        sel!("setVertexBytes:length:atIndex:")
    } else {
        sel!("setFragmentBytes:length:atIndex:")
    };
    let _: () = unsafe { msg3(
        encoder,
        sel,
        value as *const T as *const c_void,
        std::mem::size_of::<T>() as NSUInteger,
        index,
    ) };
}

fn scaled(color: crate::math::Color, intensity: f32) -> [f32; 4] {
    [
        color.r * intensity,
        color.g * intensity,
        color.b * intensity,
        0.0,
    ]
}

/// three.js's `depthFunc` as an `MTLCompareFunction` (already flipped for a
/// reverse-Z pass by the caller).
fn metal_compare(func: crate::materials::DepthFunc) -> NSUInteger {
    use crate::materials::DepthFunc;
    match func {
        DepthFunc::Never => compare::NEVER,
        DepthFunc::Always => compare::ALWAYS,
        DepthFunc::Less => compare::LESS,
        DepthFunc::LessEqual => compare::LESS_EQUAL,
        DepthFunc::Equal => compare::EQUAL,
        DepthFunc::GreaterEqual => compare::GREATER_EQUAL,
        DepthFunc::Greater => compare::GREATER,
        DepthFunc::NotEqual => compare::NOT_EQUAL,
    }
}

/// The cull mode for a caster of three.js `side` in a shadow pass. three.js's
/// `shadowSide` records a front-sided caster's back faces, a back-sided one's
/// front faces and a double-sided one's both. The point pass's cube faces
/// mirror one axis, which reverses winding, so the faces to cull swap there.
fn shadow_cull(side: u32, radial: bool) -> NSUInteger {
    let (front, back) = if radial { (cull::BACK, cull::FRONT) } else { (cull::FRONT, cull::BACK) };
    match side {
        1 => back,
        2 => cull::NONE,
        _ => front,
    }
}

/// A material draws blended when it is transparent by three.js's rule
/// ([`Material::transparent`]): its `transparent` flag, or — for a material
/// that never set one — opacity below 1.
fn is_transparent(material: &Material) -> bool {
    material.transparent()
}

/// Rotate + scale a direction by a world matrix, ignoring translation.
fn transform_direction(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

/// The inverse transpose of `m`, for transforming normals under non-uniform
/// scale. Returned as a 4x4 so the shader can read its columns directly.
pub(crate) fn normal_matrix(m: &Matrix4) -> [f32; 16] {
    let inv = m.invert();
    let e = inv.elements;
    let mut t = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            t[col * 4 + row] = e[row * 4 + col];
        }
    }
    t
}

/// A geometry's bounding sphere: its own when it has computed one, else
/// from its positions (around their box's centre, as three.js's).
fn geometry_sphere(g: &BufferGeometry) -> Option<crate::math::Sphere> {
    if let Some(s) = g.bounding_sphere {
        return Some(s);
    }
    // As wgpu's: a compute pass moves its vertices, so the CPU positions
    // give no bounds to cull by.
    if g.gpu_writable {
        return None;
    }
    let (mut lo, mut hi) = (Vector3::new(f32::MAX, f32::MAX, f32::MAX), Vector3::new(f32::MIN, f32::MIN, f32::MIN));
    let mut any = false;
    for p in g.positions()? {
        lo = Vector3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vector3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        any = true;
    }
    if !any {
        return None;
    }
    let centre = (lo + hi) * 0.5;
    let r = g.positions()?.map(|p| (p - centre).length_sq()).fold(0.0f32, f32::max).sqrt();
    Some(crate::math::Sphere::new(centre, r))
}

/// What a frame's shadow maps show, hashed: the passes' matrices and every
/// caster's geometry, transform, side, cut-out, pose, instances and morph —
/// all `encode_shadow` reads. Equal from one render to the next, the maps are
/// reused (`ShadowMap`), as the wgpu path does. A cheap word mix (FxHash's):
/// the inputs are trusted, and this runs over every caster each frame.
fn shadow_fingerprint(items: &[DrawItem]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| h = (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    let floats = |mix: &mut dyn FnMut(u64), v: &[f32]| {
        mix(v.len() as u64);
        v.iter().for_each(|x| mix(x.to_bits() as u64));
    };
    for item in items.iter().filter(|i| i.cast_shadow && i.primitive == primitive::TRIANGLE) {
        mix(Arc::as_ptr(&item.geometry) as usize as u64);
        mix(item.geometry.geometry_version as u64);
        item.world.elements.iter().for_each(|v| mix(v.to_bits() as u64));
        let m = &item.material;
        mix(m.side() as u64);
        mix(m.alpha_test().to_bits() as u64);
        let slots = m.texture_slots();
        mix(slots.map.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize as u64));
        mix(slots.alpha_map.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize as u64));
        mix(((item.range.0 as u64) << 32) | item.range.1 as u64);
        mix(item.instance_count as u64);
        floats(&mut mix, item.instances.as_deref().unwrap_or(&[]));
        floats(&mut mix, item.instance_morph.as_ref().map_or(&[][..], |(rows, _)| rows));
        floats(&mut mix, item.bones.as_deref().unwrap_or(&[]));
        floats(&mut mix, &item.morph_influences);
    }
    h
}

/// Per-draw uniforms for one item.
/// Negate clip-space x.
///
/// Cube maps are defined left-handed in every graphics API, and
/// [`Matrix4::look_at`] is right-handed — so rendering a face with it lays the
/// texels down mirrored against what the sampler will later expect. Each face
/// still looks perfectly reasonable on its own, which is what makes this hard
/// to see: the shadow simply lands on the wrong side.
///
/// *Which* axis is not a guess. `tests/shadow_fixtures.rs` measures how far the
/// shadow lands from where the geometry puts it, in pixels:
///
/// ```text
///            none      x       y      xy
///   wgpu     79.2   133.1     2.1   107.0
///   metal   107.0     2.1   133.1    79.2
/// ```
///
/// This path wants x and the wgpu one wants y, and the two tables are each
/// other's shifted by one flip — their framebuffer-y conventions are opposite,
/// so the correction is too. Anything but the 2.1 is a shadow of something
/// that is not there.
fn mirror_clip_x(m: Matrix4) -> Matrix4 {
    let mut e = m.elements;
    // Column-major: row 0 is elements 0, 4, 8, 12.
    e[0] = -e[0];
    e[4] = -e[4];
    e[8] = -e[8];
    e[12] = -e[12];
    Matrix4 { elements: e }
}

/// Cache key for an uploaded texture: the image, and which transfer curve it
/// was uploaded under. See [`MetalRenderer::ensure_texture_as`].
/// Per-vertex joint indices and weights, as the shader wants them.
///
/// Both attributes are four floats a vertex in `BufferGeometry` — three.js
/// stores joint *indices* as floats too — so this is a repack rather than a
/// conversion. Vertices past the end of either attribute are bound rigidly to
/// bone 0, which is what an unweighted vertex means.
fn build_skin(geometry: &BufferGeometry, vertex_count: usize) -> Vec<SkinVertex> {
    use crate::core::BufferAttribute;
    let joints = geometry.get_attribute("joint");
    let weights = geometry.get_attribute("weight");
    (0..vertex_count)
        .map(|i| {
            let take = |a: Option<&BufferAttribute>, fallback: [f32; 4]| -> [f32; 4] {
                match a {
                    Some(a) if (i + 1) * 4 <= a.array.len() => {
                        [a.array[i * 4], a.array[i * 4 + 1], a.array[i * 4 + 2], a.array[i * 4 + 3]]
                    }
                    _ => fallback,
                }
            };
            SkinVertex {
                joints: take(joints, [0.0; 4]),
                weights: take(weights, [1.0, 0.0, 0.0, 0.0]),
            }
        })
        .collect()
}

/// The `MorphInfo` uniform for drawing `geometry` (whose GPU morph buffer is
/// `morph`) under `influences`: count, base influence and weights from
/// [`crate::core::morph_uniforms`] — the numbers the wgpu backend uploads —
/// with this shader's flags (1 normal, 2 colour, 4 position). Count 0, no
/// blending, when nothing blends.
fn morph_uniforms(
    geometry: &BufferGeometry,
    morph: Option<&MorphBuffer>,
    influences: &[f32],
) -> MorphUniforms {
    let (Some(m), Some((header, weights))) =
        (morph, crate::core::morph_uniforms(geometry, influences))
    else {
        return MorphUniforms::default();
    };
    let mut u = MorphUniforms {
        info: [header[0] as u32, m.stride, m.flags, m.vertex_count],
        base: [header[1], 0.0, 0.0, 0.0],
        weights: [[0.0; 4]; 16],
    };
    for (i, w) in weights.iter().enumerate() {
        u.weights[i / 4][i % 4] = *w;
    }
    u
}

/// [`morph_uniforms`] for a draw item: an `InstancedMesh` with per-instance
/// rows reads its influences from them (flag 16; `base.y` the row length,
/// `base.z` where the rows start in the instance buffer, in floats).
fn item_morph_uniforms(item: &DrawItem, morph: Option<&MorphBuffer>) -> MorphUniforms {
    let Some((row_len, offset)) = item.instance_morph_layout else {
        return morph_uniforms(&item.geometry, morph, &item.morph_influences);
    };
    let (Some(m), Some(header)) = (morph, crate::core::morph_header(&item.geometry)) else {
        return MorphUniforms::default();
    };
    MorphUniforms {
        info: [header[0] as u32, m.stride, m.flags | 16, m.vertex_count],
        base: [0.0, row_len as f32, offset as f32, 0.0],
        weights: [[0.0; 4]; 16],
    }
}

fn texture_key(texture: &Arc<Texture>, linear: bool) -> usize {
    // The pointer is 16-byte aligned, so the low bit is free to carry the role.
    (Arc::as_ptr(texture) as usize) | linear as usize
}

/// The distinct samplers one draw's maps use, bound at slots 0 to
/// `SAMPLER_POOL − 1`, and each map's index among them. The base map's is
/// always slot 0, which the point and cut-out shaders read directly. A draw
/// whose maps want more distinct filter-and-wrap states than there are slots
/// reads the extra ones through the base map's.
struct SamplerPool {
    ids: Vec<Id>,
    index: [u32; MAP_COUNT],
}

impl SamplerPool {
    fn new(base: Id) -> Self {
        Self { ids: vec![base], index: [0; MAP_COUNT] }
    }

    fn set(&mut self, map: usize, sampler: Id) {
        let i = match self.ids.iter().position(|&s| s == sampler) {
            Some(i) => i,
            None if self.ids.len() < SAMPLER_POOL => {
                self.ids.push(sampler);
                self.ids.len() - 1
            }
            None => 0,
        };
        self.index[map] = i as u32;
    }

    /// Every slot, the unused ones filled with the base map's sampler:
    /// Metal requires each declared sampler to be bound.
    fn slots(&self) -> [Id; SAMPLER_POOL] {
        std::array::from_fn(|i| self.ids.get(i).copied().unwrap_or(self.ids[0]))
    }

    fn packed(&self) -> [[u32; 4]; 6] {
        std::array::from_fn(|v| std::array::from_fn(|k| self.index.get(4 * v + k).copied().unwrap_or(0)))
    }
}

fn draw_uniforms(
    item: &DrawItem,
    has_map: bool,
    instanced: bool,
    geometry: &BufferGeometry,
) -> DrawUniforms {
    let material = &item.material;
    let color = material.color();
    let emissive = material.emissive();
    let specular = material.specular();
    let (depth_near, depth_far) = material.depth_range();
    let slots = material.texture_slots();
    // A sprite's normal matrix slot carries its billboard parameters instead
    // (see `sprite_vertex` in the shader); `object.y` says so.
    let normal_mat = match item.sprite {
        Some(p) => {
            let mut m = [0.0f32; 16];
            m[..4].copy_from_slice(&p);
            m
        }
        None => normal_matrix(&item.world),
    };
    let is_sprite = item.sprite.is_some() as u32;
    DrawUniforms {
        model: item.world.elements,
        normal_mat,
        base_color: [color.r, color.g, color.b, material.opacity()],
        emissive: [emissive.r, emissive.g, emissive.b, 0.0],
        specular: [specular.r, specular.g, specular.b, material.shininess()],
        pbr: [
            material.roughness(),
            material.metalness(),
            material.alpha_test(),
            // A toon `gradientMap` rides the matcap slot (see `encode_pass`).
            material.texture_slots().gradient_map.is_some() as u32 as f32,
        ],
        extra: [0; 4],
        map_scalars: material.map_uniforms()[0],
        env_params: material.map_uniforms()[1],
        env_dir: {
            let u = material.map_uniforms();
            [u[2], u[3], u[4]]
        },
        map_xform: map_xforms(&slots),
        planet: {
            let atmos = material.atmosphere_params();
            let (eclipse, sun_radius) = material.eclipse();
            let scatter = material.scatter_params();
            let twilight = material.twilight_color();
            let cloud_map = slots.cloud_shadow_map.is_some() && atmos[1] > 0.0;
            [
                [material.emissive_night_side(), sun_radius, atmos[3], scatter[0]],
                [twilight.r, twilight.g, twilight.b, scatter[1]],
                [atmos[0], atmos[1], atmos[2], cloud_map as u32 as f32],
                eclipse,
                {
                    let (scale, bias) = material.displacement();
                    [material.normal_scale().y, scale, bias, slots.displacement_map.is_some() as u32 as f32]
                },
            ]
        },
        // Filled in at bind time, where the pool and the uploaded set are known.
        map_samplers: [[0; 4]; 6],
        phys_flags: [0; 4],
        physical: match &**material {
            Material::Physical(m) => [
                [m.ior, m.specular_intensity, m.clearcoat, m.clearcoat_roughness],
                [m.specular_color.r, m.specular_color.g, m.specular_color.b, m.sheen],
                [m.sheen_color.r, m.sheen_color.g, m.sheen_color.b, m.sheen_roughness],
                [
                    m.iridescence,
                    m.iridescence_ior,
                    m.iridescence_thickness_range[0],
                    m.iridescence_thickness_range[1],
                ],
                [m.anisotropy, m.anisotropy_rotation, m.clearcoat_normal_scale.x, m.clearcoat_normal_scale.y],
                [
                    m.transmission,
                    m.thickness,
                    if m.attenuation_distance.is_finite() { m.attenuation_distance.max(0.0) } else { 0.0 },
                    m.dispersion,
                ],
                [m.attenuation_color.r, m.attenuation_color.g, m.attenuation_color.b, 0.0],
            ],
            // The atmosphere has no physical maps; its own parameters ride
            // in these slots, as the wgpu path's `params3/4/13/14` (see
            // `KIND_ATMOSPHERE` in the shader).
            Material::Atmosphere(m) => [
                [m.planet_radius, m.atmosphere_radius, m.intensity, m.falloff],
                [m.sunset_color.r, m.sunset_color.g, m.sunset_color.b, m.airglow],
                material.scatter_params(),
                material.aurora_params(),
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ],
            _ => [[0.0; 4]; 7],
        },
        // `maps` is filled in at bind time, where the uploaded set is known.
        maps: [0; 4],
        object: [
            item.receive_shadow as u32,
            is_sprite,
            material.shadow_only() as u32,
            material.tone_mapped() as u32,
        ],
        map_params: [
            material.normal_scale().x,
            material.ao_intensity(),
            1.0,
            0.0,
        ],
        misc: [
            0.0,
            material.point_size(),
            if material.point_size_attenuation() {
                1.0
            } else {
                0.0
            },
            0.0,
        ],
        flags: [
            material.kind() as u32,
            has_map as u32,
            geometry.attributes.contains_key("color") as u32,
            // Bit 0: instanced; above it, where the instance colours start.
            if instanced { 1 | (item.instance_color_offset << 1) } else { 0 },
        ],
    }
    .with_depth_range(depth_near, depth_far)
}

impl DrawUniforms {
    /// `misc.z/w` carry the point-size attenuation flag for points and the
    /// depth range for `MeshDepthMaterial` — the two never apply at once.
    fn with_depth_range(mut self, near: f32, far: f32) -> Self {
        if self.flags[0] == MaterialKind::Depth as u32
            || self.flags[0] == MaterialKind::Distance as u32
        {
            self.misc[2] = near;
            self.misc[3] = far;
        }
        self
    }
}

/// Interleave a geometry's attributes into the shader's `Vertex` layout.
pub(crate) fn build_vertices(geometry: &BufferGeometry) -> Vec<Vertex> {
    let Some(positions) = geometry.attributes.get("position") else {
        return Vec::new();
    };
    if positions.item_size < 3 {
        return Vec::new();
    }
    let count = positions.array.len() / positions.item_size;
    let normals = geometry
        .attributes
        .get("normal")
        .filter(|a| a.item_size >= 3);
    let uvs = geometry.attributes.get("uv").filter(|a| a.item_size >= 2);
    let colors = geometry
        .attributes
        .get("color")
        .filter(|a| a.item_size >= 3);
    let derived = if normals.is_none() {
        // A mesh with no normals would light to black. Deriving them costs one
        // pass over the index buffer at upload and nothing per frame.
        Some(derive_normals(geometry, count))
    } else {
        None
    };

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let p = positions.item_size * i;
        let normal = match (normals, &derived) {
            (Some(n), _) => {
                let j = n.item_size * i;
                [n.array[j], n.array[j + 1], n.array[j + 2]]
            }
            (None, Some(d)) => d[i],
            (None, None) => [0.0, 0.0, 1.0],
        };
        let uv = uvs
            .map(|a| {
                let j = a.item_size * i;
                [a.array[j], a.array[j + 1]]
            })
            .unwrap_or([0.0, 0.0]);
        let color = colors
            .map(|a| {
                let j = a.item_size * i;
                [a.array[j], a.array[j + 1], a.array[j + 2]]
            })
            .unwrap_or([1.0, 1.0, 1.0]);
        let alpha = colors
            .filter(|a| a.item_size >= 4)
            .map(|a| a.array[a.item_size * i + 3])
            .unwrap_or(1.0);
        out.push(Vertex {
            position: [
                positions.array[p],
                positions.array[p + 1],
                positions.array[p + 2],
            ],
            normal,
            uv,
            color,
            alpha,
        });
    }
    out
}

/// Interleave a line geometry's attributes into the slim `LineVertex` layout.
///
/// Deliberately does not derive normals: lines are unlit, and `derive_normals`
/// would allocate a scratch vector the size of the geometry to compute
/// something nothing reads.
pub(crate) fn build_line_vertices(geometry: &BufferGeometry) -> Vec<LineVertex> {
    let Some(positions) = geometry.attributes.get("position") else {
        return Vec::new();
    };
    if positions.item_size < 3 {
        return Vec::new();
    }
    let count = positions.array.len() / positions.item_size;
    let colors = geometry
        .attributes
        .get("color")
        .filter(|a| a.item_size >= 3);

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let p = positions.item_size * i;
        let color = colors
            .map(|a| {
                let j = a.item_size * i;
                [a.array[j], a.array[j + 1], a.array[j + 2]]
            })
            .unwrap_or([1.0, 1.0, 1.0]);
        out.push(LineVertex {
            position: [
                positions.array[p],
                positions.array[p + 1],
                positions.array[p + 2],
            ],
            color,
        });
    }
    out
}

/// Area-weighted vertex normals, for geometry that arrived without any.
fn derive_normals(geometry: &BufferGeometry, count: usize) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; count];
    let positions = match geometry.attributes.get("position") {
        Some(p) => p,
        None => return normals,
    };
    let at = |i: usize| {
        let j = positions.item_size * i;
        Vector3::new(
            positions.array[j],
            positions.array[j + 1],
            positions.array[j + 2],
        )
    };
    let mut accumulate = |a: usize, b: usize, c: usize| {
        if a >= count || b >= count || c >= count {
            return;
        }
        // Un-normalised cross product: its length is twice the triangle's
        // area, which is exactly the weight a smooth normal wants.
        let n = (at(b) - at(a)).cross(at(c) - at(a));
        for i in [a, b, c] {
            normals[i][0] += n.x;
            normals[i][1] += n.y;
            normals[i][2] += n.z;
        }
    };
    match &geometry.index {
        Some(idx) => {
            for tri in idx.as_chunks::<3>().0 {
                accumulate(tri[0] as usize, tri[1] as usize, tri[2] as usize);
            }
        }
        None => {
            for t in 0..count / 3 {
                accumulate(t * 3, t * 3 + 1, t * 3 + 2);
            }
        }
    }
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-12 {
            *n = [n[0] / len, n[1] / len, n[2] / len];
        } else {
            *n = [0.0, 0.0, 1.0];
        }
    }
    normals
}

/// The quad a `Sprite` draws.
///
/// A `Sprite` carries a material and no geometry, so the backend supplies one:
/// a shared unit quad on the XY plane, corners at ±0.5 with three.js's sprite
/// uvs. `vs_mesh` turns it into the camera-facing quad three.js's
/// `sprite_vert` draws, from the draw's `sprite` parameters.
fn sprite_geometry(_sprite: &crate::core::Sprite) -> Option<Arc<BufferGeometry>> {
    static QUAD: std::sync::OnceLock<Arc<BufferGeometry>> = std::sync::OnceLock::new();
    Some(
        QUAD.get_or_init(|| Arc::new(crate::geometries::PlaneGeometry::new(1.0, 1.0)))
            .clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BufferAttribute;

    #[test]
    fn uniform_structs_match_the_shader() {
        // Sizes are the ABI contract with `shaders.metal`.
        assert_eq!(std::mem::size_of::<Vertex>(), 48);
        assert_eq!(std::mem::size_of::<DirLightGpu>(), 32);
        assert_eq!(std::mem::size_of::<PointLightGpu>(), 48);
        assert_eq!(std::mem::size_of::<SpotLightGpu>(), 64);
        assert_eq!(std::mem::size_of::<HemiLightGpu>(), 48);
        assert_eq!(
            std::mem::size_of::<DrawUniforms>(),
            // `model` and `normal_mat`; nine float4/uint4 rows; `extra`,
            // `map_scalars`, `env_params` and the three `env_dir` rows; the
            // six `map_samplers`, seven `physical` and one `phys_flags` rows;
            // two `map_xform` rows a map; and the five `planet` rows — the
            // members `struct Draw` declares in `shaders.metal`.
            64 + 64 + 9 * 16 + 6 * 16 + 6 * 16 + 7 * 16 + 16 + 2 * MAP_COUNT * 16 + 5 * 16
        );
        assert_eq!(
            std::mem::size_of::<FrameUniforms>(),
            // Per view: view_proj, view, camera_pos. Then 6 float4s of frame
            // state, the shadow pass's matrices and point light, the light
            // arrays, the rect-area lights, and each light's shadow.
            MAX_VIEWS * (64 + 64 + 16)
                + 16 * 6
                + MAX_DIR_LIGHTS * 32
                + MAX_POINT_LIGHTS * 48
                + MAX_SPOT_LIGHTS * 64
                + MAX_HEMI_LIGHTS * 48
                + 16 + MAX_RECT_LIGHTS * 64 // rect_counts, rects
                + 16 // shadow_info
                + MAX_SHADOW_RECORDS * (4 * 16 + 4 * 64) // shadow records
        );
        assert_eq!(std::mem::size_of::<ShadowPassGpu>(), 64 + 16);
        // The frame goes in a buffer (it is past `setVertexBytes`'s 4 KB);
        // the draw uniforms still go inline.
        assert!(std::mem::size_of::<DrawUniforms>() <= 4096);
    }

    fn triangle() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g
    }

    #[test]
    fn missing_normals_are_derived() {
        let verts = build_vertices(&triangle());
        assert_eq!(verts.len(), 3);
        // A CCW triangle on the XY plane faces +Z.
        for v in &verts {
            assert!((v.normal[2] - 1.0).abs() < 1e-5, "{:?}", v.normal);
        }
    }

    #[test]
    fn attributes_interleave_in_order() {
        let mut g = triangle();
        g.set_attribute(
            "normal",
            BufferAttribute::new(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.set_attribute(
            "uv",
            BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2),
        );
        g.set_attribute(
            "color",
            BufferAttribute::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3),
        );
        let verts = build_vertices(&g);
        assert_eq!(verts[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(verts[1].normal, [0.0, 1.0, 0.0]);
        assert_eq!(verts[1].uv, [1.0, 0.0]);
        assert_eq!(verts[2].color, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn geometry_without_positions_is_skipped() {
        assert!(build_vertices(&BufferGeometry::new()).is_empty());
        assert!(build_line_vertices(&BufferGeometry::new()).is_empty());
    }

    #[test]
    fn line_vertices_are_half_the_size_and_carry_position_and_colour() {
        // The whole point of the slim layout. If this ever stops holding, the
        // connectome renders go back to paging.
        assert_eq!(std::mem::size_of::<LineVertex>(), 24);
        assert_eq!(std::mem::size_of::<Vertex>(), 48);

        let mut g = triangle();
        g.set_attribute(
            "color",
            BufferAttribute::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3),
        );
        let verts = build_line_vertices(&g);
        assert_eq!(verts.len(), 3);
        assert_eq!(verts[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(verts[2].color, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn line_vertices_default_to_white_without_a_colour_attribute() {
        // Matches `build_vertices`, whose shader multiplies by this when the
        // material does not enable vertex colours.
        let verts = build_line_vertices(&triangle());
        assert_eq!(verts[0].color, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn normal_matrix_is_inverse_transpose() {
        let m = Matrix4::compose(
            Vector3::new(1.0, 2.0, 3.0),
            crate::math::Quaternion::identity(),
            Vector3::new(2.0, 1.0, 0.5),
        );
        let n = normal_matrix(&m);
        // Under a non-uniform scale, a normal along x scales by 1/2, not 2.
        assert!((n[0] - 0.5).abs() < 1e-5, "{}", n[0]);
        assert!((n[5] - 1.0).abs() < 1e-5, "{}", n[5]);
        assert!((n[10] - 2.0).abs() < 1e-5, "{}", n[10]);
    }

    #[test]
    fn transparency_follows_three_js() {
        use crate::materials::{BasicMaterial, StandardMaterial};
        use crate::math::Color;
        let opaque = Material::Basic(BasicMaterial::default());
        assert!(!is_transparent(&opaque));
        // three.js blends only a material marked `transparent`: a Basic
        // material at opacity 0.5 without the flag draws opaquely, as wgpu
        // draws it.
        let faded = Material::Basic(BasicMaterial::default().with_opacity(0.5));
        assert!(!is_transparent(&faded));
        let flagged = Material::Basic(BasicMaterial::default().with_transparent(true));
        assert!(is_transparent(&flagged));
        // A native material that never set the flag keeps blending below
        // opacity 1 (`transparent: None`); setting it decides.
        let mut lit = StandardMaterial::new(Color::WHITE);
        lit.opacity = 0.5;
        assert!(is_transparent(&Material::Standard(lit.clone())));
        lit.transparent = Some(false);
        assert!(!is_transparent(&Material::Standard(lit)));
    }
}
