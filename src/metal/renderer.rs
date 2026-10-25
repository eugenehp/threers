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
}

/// How many material maps a draw can bind, `MAP_*` in `shaders.metal`.
const MAP_COUNT: usize = 22;
/// Where the physical maps start, in that order (`MAP_PHYS0`), and at which
/// texture slot they are bound.
const MAP_PHYS0: usize = 10;
const PHYS_TEXTURE0: usize = 16;
/// Sampler slots 0–9: one per distinct filter-and-wrap among a draw's maps.
const SAMPLER_POOL: usize = 10;

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
    /// Depth-stencil states by (`MTLCompareFunction`, write), made on first
    /// use: every material picks its own from `depthTest` / `depthWrite` /
    /// `depthFunc`.
    depth_states: HashMap<(NSUInteger, bool), Owned>,
    white: Owned,
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
        let ltc = crate::metal::resources::upload_ltc(&device)?;
        let capture_dummy = crate::metal::resources::black_texture_array(&device)?;
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
            depth_states,
            white,
            ltc,
            capture_dummy,
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

    pub fn set_cache_retention(&mut self, frames: u64) {
        self.cache_retention = frames;
    }

    /// Draw `scene` from `camera` into `pass`, on a command buffer of its own.
    ///
    /// Does not block: the command buffer is committed and the call returns. A
    /// readback or a present sequences behind it on the same queue.
    pub fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        pass: &PassAttachments,
    ) -> Result<MetalRenderStats, MetalError> {
        self.render_views(scene, &[RenderView::from_camera(camera)], pass)
    }

    /// Draw `scene` once for each view — one for a window, two for the eyes of
    /// a headset.
    ///
    /// Both eyes go through the scene graph, the sort and the encoder **once**
    /// when the views are *layered*: same colour texture, same viewport,
    /// different array slices. The vertex stage then reads its eye out of the
    /// instance id and writes `render_target_array_index`, so the second eye
    /// costs pixels and not CPU. Anything else — a texture or a viewport per
    /// eye — is one pass per view, and the clear happens once per distinct
    /// target so the second pass cannot wipe the first.
    ///
    /// At most [`MAX_VIEWS`] views; extra ones are ignored.
    pub fn render_views(
        &mut self,
        scene: &mut Scene,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> Result<MetalRenderStats, MetalError> {
        let _pool = AutoreleasePool::new();
        if views.is_empty() {
            return Err(MetalError::Unsupported("render with no views".into()));
        }
        let views = &views[..views.len().min(MAX_VIEWS)];
        // The viewport / scissor / auto-clear state. The default state plans
        // as `legacy` and takes the path below untouched.
        self.viewport_state.buffer_size = (pass.width, pass.height);
        let staged = match self.viewport_state.settings.plan(pass.width, pass.height) {
            None => {
                // Nothing visible: draw nothing, but still present a frame.
                self.present(pass)?;
                return Ok(MetalRenderStats::default());
            }
            Some(p) if p.legacy => None,
            Some(p) => Some(self.viewport_state.stage(&p, views, pass)?),
        };
        // A pane drawn into its own target is copied back afterwards.
        let composite = staged.as_ref().and_then(|s| s.composite);
        let (views, pass): (&[RenderView], &PassAttachments) = match &staged {
            Some(s) => (&s.views, &s.pass),
            None => (views, pass),
        };
        self.frame = self.frame.wrapping_add(1);
        let slot = (self.frame % FRAMES_IN_FLIGHT as u64) as usize;
        self.transients[slot].clear();

        scene.update_world();
        self.ensure_environment(scene);
        let (mut frame_u, mut items, mut stats, casters) = self.collect(scene, views, pass);
        // Each casting light's record — layers, matrices, biases, filter —
        // and the passes that draw its maps, as the wgpu renderer builds them.
        self.ensure_shadow_arrays(&casters)?;
        // `(kind: 0 directional, 1 spot, 2 point; layer; matrix; point light)`.
        let mut shadow_passes: Vec<(u8, usize, [f32; 16], [f32; 4])> = Vec::new();
        let mut next_layer = [0usize; 3];
        for (r, caster) in casters.iter().enumerate() {
            let settings = caster.settings();
            let filter = settings.filter.unwrap_or(self.shadow_type).code();
            let (bias, normal_bias) = (settings.bias.max(0.0), settings.normal_bias.max(0.0));
            let mut rec = ShadowRecordGpu::default();
            match *caster {
                Caster::Dir { dir, settings } => {
                    let texel = 1.0 / self.shadow_size.max(1) as f32;
                    let layer = next_layer[0];
                    let cascades = caster.layers();
                    next_layer[0] += cascades;
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
                        let view = &views[0];
                        for (c, (vp, far)) in crate::lights::shadow::cascade_frusta(
                            &view.view_matrix,
                            &view.projection_matrix,
                            dir,
                            &settings,
                            self.shadow_size,
                        )
                        .into_iter()
                        .enumerate()
                        {
                            rec.vp[c] = vp;
                            rec.extra[c] = far;
                            shadow_passes.push((0, layer + c, vp, [0.0; 4]));
                        }
                    } else {
                        let s = settings.camera_size;
                        // A fixed cube on the origin, matching the wgpu path.
                        let eye = dir.normalize() * (-s * 1.5);
                        let view = Matrix4::look_at(eye, Vector3::ZERO, Vector3::UP);
                        let proj =
                            Matrix4::orthographic(-s, s, s, -s, settings.camera_near, settings.camera_far);
                        let vp = proj.multiply(&view).elements;
                        rec.vp[0] = vp;
                        rec.extra = [f32::MAX; 4];
                        shadow_passes.push((0, layer, vp, [0.0; 4]));
                    }
                }
                Caster::Spot { pos, dir, angle, settings } => {
                    let texel = 1.0 / self.spot_shadow_size.max(1) as f32;
                    let layer = next_layer[1];
                    next_layer[1] += 1;
                    let view = Matrix4::look_at(pos, pos + dir.normalize(), Vector3::UP);
                    // fov = 2 × the outer angle, so the frustum covers the whole cone.
                    let fov = (angle * 2.0).clamp(0.05, std::f32::consts::PI - 0.05);
                    let proj =
                        Matrix4::perspective(fov, 1.0, settings.camera_near, settings.camera_far);
                    let vp = proj.multiply(&view).elements;
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    let scale = size / (2.0 * (fov * 0.5).tan());
                    rec.info = [1, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.05).clamp(2.0 * texel, 0.1), settings.camera_near, settings.camera_far, scale];
                    rec.vp[0] = vp;
                    shadow_passes.push((1, layer, vp, [0.0; 4]));
                }
                Caster::Point { pos, radius, settings } => {
                    // A face texel (`2 / size` of a unit cube face) per unit
                    // of radius, across the direction to the light.
                    let texel = 2.0 / POINT_SHADOW_SIZE as f32;
                    let k = next_layer[2];
                    next_layer[2] += 6;
                    let at = [pos.x, pos.y, pos.z, radius];
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    rec.info = [2, k as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.extra = at;
                    rec.pcss = [(size * 0.05).clamp(2.0 * texel, 0.2), 0.0, 0.0, size];
                    // Six faces, +X −X +Y −Y +Z −Z, each a 90° frustum. Metal's
                    // cube convention, which is not the GL one the wgpu path
                    // uses: +Y is up on the side faces rather than −Y. Getting
                    // this wrong leaves each face individually plausible and
                    // the shadow landing in the wrong place.
                    let faces: [(Vector3, Vector3); 6] = [
                        (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)),
                        (Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)),
                        (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 1.0, 0.0)),
                    ];
                    let proj = Matrix4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, radius.max(1.0));
                    for (face, &(forward, up)) in faces.iter().enumerate() {
                        let view = Matrix4::look_at(pos, pos + forward, up);
                        let vp = mirror_clip_x(proj.multiply(&view)).elements;
                        shadow_passes.push((2, k + face, vp, at));
                    }
                }
            }
            frame_u.shadows[r] = rec;
        }
        frame_u.shadow_info = [casters.len() as u32, 0, 0, 0];
        // What the maps show, for reusing them (`ShadowMap`): taken here,
        // before the uploads below move the instance and bone data out.
        let shadow_content = (!shadow_passes.is_empty()).then(|| shadow_fingerprint(&items));

        // Opaque front-to-back (early-z rejects the overdraw), transparent
        // back-to-front (blending is order-dependent). `render_order` wins over
        // both, as in three.js.
        items.sort_by(|a, b| {
            a.transparent
                .cmp(&b.transparent)
                // three.js's transmissive list sits between the two.
                .then(a.transmissive.cmp(&b.transmissive))
                .then(a.render_order.cmp(&b.render_order))
                .then_with(|| {
                    let (x, y) = if a.transparent {
                        (b.view_depth, a.view_depth)
                    } else {
                        (a.view_depth, b.view_depth)
                    };
                    x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Upload before encoding: both touch `self`, and the encoder holds a
        // command buffer that a mipmap blit would otherwise interleave with.
        for item in &mut items {
            let slim = item.primitive == primitive::LINE && slim_lines_enabled();
            if self.ensure_geometry(&item.geometry, slim)? {
                stats.geometry_uploads += 1;
            }
            if item.primitive == primitive::TRIANGLE && item.material.wireframe() {
                let full = item.range == (0, item.geometry.draw_count() as u32);
                if full {
                    self.ensure_wire_indices(&item.geometry)?;
                } else {
                    // A group or a draw range: the line list of that span
                    // alone, for this frame.
                    let flat = item.geometry.wireframe_indices_in(item.range.0, item.range.1);
                    if !flat.is_empty() {
                        let buf = self.device.new_buffer(bytemuck::cast_slice(&flat))?;
                        item.wire_span = Some((self.transients[slot].len(), flat.len()));
                        self.transients[slot].push(buf);
                    }
                }
            }
            let slots = item.material.texture_slots();
            // Every slot the fragment shader can bind. Uploading only the ones
            // it used to sample is how the extra maps stayed invisible after
            // the shader learned to read them: the binding asked the cache for
            // a texture nobody had put there.
            let phys: Vec<(Arc<Texture>, bool)> = physical_maps(&slots)
                .into_iter()
                .filter_map(|(m, l)| m.map(|m| (m.clone(), l)))
                .collect();
            for (map, linear) in [
                (slots.map, false),
                (slots.matcap_map, false),
                (slots.gradient_map, false),
                (slots.emissive_map, false),
                (slots.normal_map, true),
                (slots.roughness_map, true),
                (slots.metalness_map, true),
                (slots.ao_map, true),
                (slots.alpha_map, true),
                // Colour, decoded by its texture's colour space, as three.js does.
                (slots.light_map, false),
                (slots.bump_map, true),
                (slots.specular_map, true),
            ]
            .into_iter()
            .filter_map(|(m, l)| m.map(|m| (m, l)))
            .chain(phys)
            {
                if self.ensure_texture_as(&map, linear)? {
                    stats.texture_uploads += 1;
                }
            }
            // A Standard/Physical material's own `envMap` prefiltered, as the
            // scene's is; a Basic/Lambert/Phong one sampled as it is.
            let pbr = slots.pbr_env_map.as_ref().map(|c| self.prefiltered(c));
            for cube in [slots.env_map.as_ref(), pbr.as_ref()].into_iter().flatten() {
                let key = crate::renderer::gpu_texture::cube_cache_key(cube);
                if !self.material_envs.contains_key(&key) {
                    if let Ok(env) = EnvGpu::upload(&self.device, cube) {
                        self.material_envs.insert(key, env);
                        stats.texture_uploads += 1;
                    }
                    let sk = SamplerKey::from_cube(cube);
                    if !self.samplers.contains_key(&sk) {
                        if let Ok(smp) = new_sampler(&self.device, sk) {
                            self.samplers.insert(sk, smp);
                        }
                    }
                }
            }
            if let Some(mut data) = item.instances.take() {
                item.instance_count = data.len() / 16;
                if let Some((rows, row_len)) = item.instance_morph.take() {
                    item.instance_morph_layout = Some((row_len as u32, data.len() as u32));
                    data.extend_from_slice(&rows);
                }
                if let Some(colors) = item.instance_colors.take() {
                    // float4-aligned, after everything else.
                    data.resize(data.len().div_ceil(4) * 4, 0.0);
                    item.instance_color_offset = (data.len() / 4) as u32;
                    data.extend_from_slice(&colors);
                }
                let buf = self.device.new_buffer(bytemuck::cast_slice(&data))?;
                item.instance_slot = Some(self.transients[slot].len());
                self.transients[slot].push(buf);
            }
            if let Some(data) = item.bones.take() {
                // Transient rather than cached: a pose changes every frame, so
                // there is nothing to reuse and a cache would only be a thing
                // to invalidate.
                if !data.is_empty() {
                    let buf = self.device.new_buffer(bytemuck::cast_slice(&data))?;
                    item.bone_slot = Some(self.transients[slot].len());
                    self.transients[slot].push(buf);
                }
            }
        }

        // Whether this render redraws them — see `ShadowMap`. They depend on
        // lights and casters, not the camera, so a render that skips them
        // samples what an identical redraw would have written: with
        // `auto_update` on, a render whose casters and lights match the last
        // one's keeps its maps.
        // Each pass's own content: the casters and its light's matrix. Only
        // the passes whose changed are redrawn — a cascade following the
        // camera does not take a still point light's faces with it.
        let sig = (next_layer[0], next_layer[1], next_layer[2], self.shadow_size, self.spot_shadow_size);
        let valid = self.shadow_sig == Some(sig);
        let pass_hashes: Vec<Option<u64>> = shadow_passes
            .iter()
            .map(|(kind, layer, vp, at)| {
                shadow_content.map(|c| {
                    let mut h = (c.rotate_left(5) ^ (*layer as u64 | (*kind as u64) << 32)).wrapping_mul(0x517c_c1b7_2722_0a95);
                    for v in vp.iter().chain(at) {
                        h = (h.rotate_left(5) ^ v.to_bits() as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                    }
                    h
                })
            })
            .collect();
        let changed: Vec<bool> = pass_hashes
            .iter()
            .enumerate()
            .map(|(k, h)| !valid || self.shadow_pass_hashes.get(k) != Some(h))
            .collect();
        let update_shadows = !shadow_passes.is_empty() && {
            self.shadow_sig = Some(sig);
            self.shadow_flags.take_update(valid, !changed.iter().any(|&c| c))
        };
        // A forced update (`needs_update` with `auto_update` off) redraws
        // every pass; an automatic one, the ones that changed.
        let draw_pass: Vec<bool> = changed
            .iter()
            .map(|&c| update_shadows && (!self.shadow_flags.auto_update || c))
            .collect();
        if update_shadows {
            self.shadow_pass_hashes = pass_hashes;
        }
        if draw_pass.iter().any(|&d| d) {
            self.shadow_updates += 1;
        }
        // Shadow maps first: the main pass samples them, so they have to be
        // filled before anything reads them. Each pass writes its own matrix
        // into `shadow_vp` and shares one depth-only vertex function, rather
        // than having three that differ only in which matrix they read.
        for (&(kind, layer, vp, at), _) in shadow_passes.iter().zip(&draw_pass).filter(|(_, &d)| d) {
            let map = match kind {
                0 => self.shadow_map.as_ref(),
                1 => self.spot_shadow_map.as_ref(),
                _ => self.point_shadow_map.as_ref(),
            };
            let Some(map) = map.map(|t| t.id()) else { continue };
            let pass_u = ShadowPassGpu { shadow_vp: vp, point_shadow_pos: at };
            // Casters outside this light's frustum are not drawn into it.
            let frustum = crate::math::Frustum::from_projection_matrix(&Matrix4 { elements: vp });
            unsafe { self.encode_shadow(&pass_u, &items, slot, map, layer, kind == 2, &frustum) }?;
        }
        // three.js's `renderTransmissionPass`, when anything is transmissive.
        if items.iter().any(|i| i.transmissive) {
            self.capture_transmission(pass, views, &frame_u, &mut items, scene, slot, &mut stats)?;
        } else {
            self.transmission = None;
            self.capture_bound = None;
        }
        frame_u.rect_counts[1] = self.transmission.as_ref().map_or(0, |c| c.levels);
        let vp = staged.as_ref().map(|s| (s.clear, s.scissor));
        unsafe { self.encode(pass, views, &frame_u, &items, scene, slot, &mut stats, vp) }?;
        if let Some(c) = composite {
            self.viewport_state.composite(&c)?;
            // The pane pass had the drawable stripped; present after the copy.
            self.present(&c.dest)?;
        }
        self.evict_unused();
        Ok(stats)
    }

    /// three.js r165's `renderTransmissionPass`: the scene without its
    /// transmissive items drawn linear and untone-mapped into a half-float
    /// target the size of this pass, from the first view, copied into a
    /// mipmapped texture whose mips Metal generates; then the back faces of
    /// double-sided transmissive items drawn over it, reading that copy, and
    /// copied again. The colour pass binds the result (`capture_bound`).
    #[allow(clippy::too_many_arguments)]
    fn capture_transmission(
        &mut self,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &mut [DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
    ) -> Result<(), MetalError> {
        // The size the views draw at: the pass, or the largest of their
        // viewports — three.js sizes the capture to the drawing buffer a view
        // renders into, and a side-by-side eye renders into half of one.
        let (w, h) = views.iter().fold((0u32, 0u32), |(w, h), v| match v.viewport {
            Some(vp) => (w.max(vp.width.round() as u32), h.max(vp.height.round() as u32)),
            None => (w.max(pass.width), h.max(pass.height)),
        });
        let (w, h) = (w.clamp(1, pass.width.max(1)), h.clamp(1, pass.height.max(1)));
        let slices = views.len().clamp(1, MAX_VIEWS) as u32;
        if self.transmission.as_ref().map(|c| (c.size, c.slices)) != Some(((w, h), slices)) {
            let levels = 32 - w.max(h).leading_zeros();
            let target = if slices > 1 {
                crate::metal::MetalRenderTarget::layered(&self.device, w, h, pixel_format::RGBA16_FLOAT, slices)?
            } else {
                crate::metal::MetalRenderTarget::new(&self.device, w, h, pixel_format::RGBA16_FLOAT, 1)?
            };
            let mipped = crate::metal::resources::new_texture_array_levels(
                &self.device,
                w,
                h,
                pixel_format::RGBA16_FLOAT,
                texture_usage::SHADER_READ | texture_usage::RENDER_TARGET,
                slices,
                levels,
            )?;
            self.transmission = Some(MetalCapture { target, mipped, size: (w, h), levels, slices });
        }
        let attachments = self.transmission.as_ref().expect("just made").target.attachments();
        // Every view captures the scene from where it stands, into its own
        // slice, over the whole capture — a view drawn into a viewport reads
        // its slice through its own projection, so where its rectangle sits
        // does not matter here.
        let capture_views: Vec<RenderView> = views[..slices as usize]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut view = v.clone();
                view.color = NIL;
                view.depth = NIL;
                view.slice = i as u32;
                view.viewport = None;
                view
            })
            .collect();
        // Linear, untone-mapped, as three.js draws the transmission target.
        let mut fu = *frame_u;
        fu.view_info[0] = slices;
        fu.view_info[1] = 0;
        fu.view_info[2] = 0;
        fu.rect_counts[1] = 1;
        fu.viewport = [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32];

        for it in items.iter_mut() {
            it.capture_skip = it.transmissive;
        }
        self.capture_bound = None;
        unsafe { self.encode(&attachments, &capture_views, &fu, items, scene, slot, stats, None) }?;
        self.copy_capture()?;

        if items.iter().any(|i| i.transmissive && i.material.side() == 2) {
            for it in items.iter_mut() {
                let back = it.transmissive && it.material.side() == 2;
                it.capture_skip = !back;
                it.side_override = back.then_some(1);
            }
            let cap = self.transmission.as_ref().expect("made above");
            self.capture_bound = Some(cap.mipped.id());
            fu.rect_counts[1] = cap.levels;
            let keep = Some((crate::renderer::viewport::ClearMode::None, None));
            unsafe { self.encode(&attachments, &capture_views, &fu, items, scene, slot, stats, keep) }?;
            self.copy_capture()?;
        }
        for it in items.iter_mut() {
            it.capture_skip = false;
            it.side_override = None;
        }
        self.capture_bound = self.transmission.as_ref().map(|c| c.mipped.id());
        Ok(())
    }

    /// The capture target's colour into level 0 of `mipped`, then its mips.
    fn copy_capture(&self) -> Result<(), MetalError> {
        let Some(cap) = &self.transmission else { return Ok(()) };
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
            let (w, h) = cap.size;
            for slice in 0..cap.slices as NSUInteger {
                let _: () = super::objc::msg9(
                    blit,
                    sel!("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
                    cap.target.color_texture(),
                    slice,
                    0usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: w as NSUInteger, height: h as NSUInteger, depth: 1 },
                    cap.mipped.id(),
                    slice,
                    0usize,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
            let _: () = msg1(blit, sel!("generateMipmapsForTexture:"), cap.mipped.id());
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    /// Drop cache entries no frame has touched in [`cache_retention`](Self::set_cache_retention)
    /// frames.
    fn evict_unused(&mut self) {
        if self.cache_retention == u64::MAX {
            return;
        }
        let cutoff = self.frame.saturating_sub(self.cache_retention);
        self.geometries.retain(|_, g| g.last_used >= cutoff);
        self.textures.retain(|_, t| t.last_used >= cutoff);
    }

    // ------------------------------------------------------------ collection

    fn collect(
        &self,
        scene: &Scene,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> (FrameUniforms, Vec<DrawItem>, MetalRenderStats, Casters) {
        // View 0 is the one the traversal uses: its layer mask decides what is
        // drawn and its position orders the transparent items. Two eyes a few
        // centimetres apart do not disagree about either.
        let primary = &views[0];
        let eye = primary.position;
        // A perspective projection's last column is (0, 0, -1, 0) — the test
        // the raycaster and the wgpu sprite shader use too.
        let primary_perspective = primary.projection_matrix.elements[15] == 0.0;
        let mut view_proj = [Matrix4::identity().elements; MAX_VIEWS];
        let mut view_matrices = [Matrix4::identity().elements; MAX_VIEWS];
        let mut camera_pos = [[0.0f32; 4]; MAX_VIEWS];
        for (i, v) in views.iter().enumerate() {
            view_proj[i] = v.projection_matrix.multiply(&v.view_matrix).elements;
            view_matrices[i] = v.view_matrix.elements;
            camera_pos[i] = [v.position.x, v.position.y, v.position.z, 1.0];
        }
        let mut frame_u = FrameUniforms {
            view_proj,
            view: view_matrices,
            camera_pos,
            // y: the target encodes sRGB on write, which data outputs (the
            // normal material) undo so the stored byte is the value.
            // z, w: the tone mapping mode and its exposure's bits.
            view_info: [
                views.len() as u32,
                pixel_format::is_srgb(pass.color_format) as u32,
                self.tone_mapping.0.shader_mode() as u32,
                self.tone_mapping.1.to_bits(),
            ],
            viewport: [
                pass.width as f32,
                pass.height as f32,
                1.0 / pass.width.max(1) as f32,
                1.0 / pass.height.max(1) as f32,
            ],
            fog_color: [
                scene.fog.color.r,
                scene.fog.color.g,
                scene.fog.color.b,
                scene.fog.mode as f32,
            ],
            fog_params: [scene.fog.near, scene.fog.far, scene.fog.density, 0.0],
            ..Default::default()
        };
        frame_u.rect_counts[2] = self.reflective_transparency as u32;

        let mut items = Vec::new();
        let mut stats = MetalRenderStats::default();
        let (mut n_dir, mut n_point, mut n_spot, mut n_hemi) = (0usize, 0, 0, 0);
        // Every light that casts, each owning a layer of its kind's maps.
        let mut casters = Casters::new();
        // Union every view's layer mask so split-screen / multi-viewport passes
        // collect both sides; each pass filters in [`encode_pass`].
        let cam_layers = views.iter().fold(Layers::with_mask(0), |mut acc, v| {
            acc.mask |= v.layers.mask;
            acc
        });
        let cam_layers = if cam_layers.mask == 0 {
            primary.layers
        } else {
            cam_layers
        };

        scene.arena.traverse_visible(scene.root, &mut |_id, obj| {
            if !cam_layers.test(&obj.layers) {
                return;
            }
            let world = obj.matrix_world;
            let obj_layers = obj.layers;
            // The instance colours for the next item pushed (instanced meshes
            // only), taken by `push_skinned`.
            let pending_colors: std::cell::RefCell<Option<Vec<f32>>> =
                std::cell::RefCell::new(None);
            #[allow(clippy::too_many_arguments)]
            let mut push_skinned = |geometry: &Arc<BufferGeometry>,
                            material: &Arc<Material>,
                            primitive: NSUInteger,
                            instances: Option<Vec<f32>>,
                            bones: Option<Vec<f32>>,
                            sprite: Option<[f32; 4]>,
                            range: Option<(u32, u32)>,
                            influences: &[f32],
                            instance_morph: Option<(Vec<f32>, usize)>| {
                if !geometry.attributes.contains_key("position") {
                    stats.skipped += 1;
                    return;
                }
                // An empty span draws nothing, as three.js's renderBufferDirect.
                let Some(range) = range else {
                    return;
                };
                let center =
                    Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
                // three.js sorts on `transparent` alone: an opaque
                // `depthTest: false` material draws in the opaque list, in
                // `render_order`, like any other.
                let transparent = is_transparent(material);
                // A transmissive physical material's glass mode.
                let glass_mode = match &**material {
                    Material::Physical(m) if m.transmission > 0.0 => Some(m.transparency),
                    _ => None,
                };
                items.push(DrawItem {
                    fur: match &**material {
                        Material::Shader(m) if crate::fur::is_fur(m) => {
                            let mut d = [0.0f32; 64];
                            for (slot, v) in m.data.iter().take(16).enumerate() {
                                d[slot * 4..slot * 4 + 4].copy_from_slice(v);
                            }
                            Some(d)
                        }
                        _ => None,
                    },
                    fx_surface: match &**material {
                        Material::Shader(m) if crate::fx::surface::is_surface(m) => {
                            Some(crate::fx::surface::metal_bytes(m))
                        }
                        _ => None,
                    },
                    range,
                    morph_influences: influences.to_vec(),
                    wire_span: None,
                    bones,
                    bone_slot: None,
                    // three.js's shadow pass renders meshes, lines and points;
                    // a sprite casts nothing.
                    cast_shadow: obj.cast_shadow && sprite.is_none(),
                    sprite,
                    receive_shadow: obj.receive_shadow,
                    geometry: geometry.clone(),
                    material: material.clone(),
                    world,
                    layers: obj_layers,
                    primitive,
                    instance_count: instances.as_ref().map(|v| v.len() / 16).unwrap_or(1),
                    instances,
                    instance_slot: None,
                    instance_morph,
                    instance_morph_layout: None,
                    instance_colors: pending_colors.borrow_mut().take(),
                    instance_color_offset: 0,
                    view_depth: center.distance_to(eye),
                    render_order: obj.render_order,
                    // `Refract` draws as three.js's transmission here: the same
                    // volume model over the capture, which is what that mode
                    // refines on the wgpu path.
                    transparent: transparent && glass_mode != Some(crate::materials::TransparencyMode::Refract),
                    transmissive: material.transmissive()
                        || glass_mode == Some(crate::materials::TransparencyMode::Refract),
                    // `Glass` and `Oit`: glass blended over what is behind,
                    // tinted by channel.
                    blended_glass: matches!(
                        glass_mode,
                        Some(crate::materials::TransparencyMode::Glass | crate::materials::TransparencyMode::Oit)
                    ),
                    // Order-independent: drawn in the accumulation pass rather
                    // than sorted with the transparent draws (`encode_oit`).
                    oit: transparent
                        && matches!(&**material, Material::Physical(m) if m.transparency == crate::materials::TransparencyMode::Oit),
                    capture_skip: false,
                    side_override: None,
                    state: material.render_state(),
                });
            };
            // Everything that is not a skinned mesh, which is everything
            // else: same closure, no pose.
            // The whole draw range, one material: lines, points, instances.
            let mut push =
                |g: &Arc<BufferGeometry>,
                 m: &Arc<Material>,
                 p: NSUInteger,
                 inst: Option<Vec<f32>>| {
                    push_skinned(g, m, p, inst, None, None, g.draw_span(None), &[], None)
                };

            match &obj.kind {
                // One item per draw — per group with a material array — each
                // with its own material, transparency and sort, as three.js's
                // render list holds them.
                ObjectKind::Mesh(m) => {
                    // No positions: counted once, as any undrawable item is
                    // (its draw list is empty, so the loop would not see it).
                    if !m.geometry.attributes.contains_key("position") {
                        push_skinned(&m.geometry, &m.material, primitive::TRIANGLE, None, None, None, None, &[], None);
                        return;
                    }
                    for d in m.draws() {
                        push_skinned(
                            &m.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            None,
                            None,
                            None,
                            Some((d.start, d.end)),
                            &m.morph_influences,
                            None,
                        );
                    }
                }
                ObjectKind::SkinnedMesh(sm) => {
                    // `bone_matrices` is world × inverse-bind per bone, already
                    // composed by `Skeleton::update`, so the vertex stage only
                    // has to blend them.
                    //
                    // Exactly `MAX_BONES` of them, padded with identities, as
                    // the wgpu backend uploads them: `vs_skinned` clamps joint
                    // indices to the last, so every index reads the matrix it
                    // reads there (and in the raycaster) — never past the end.
                    let identity = Matrix4::identity();
                    let bones: Vec<f32> = (0..crate::core::MAX_BONES)
                        .flat_map(|i| {
                            sm.skeleton
                                .bone_matrices
                                .get(i)
                                .unwrap_or(&identity)
                                .elements
                        })
                        .collect();
                    for d in sm.draws() {
                        push_skinned(
                            &sm.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            None,
                            Some(bones.clone()),
                            None,
                            Some((d.start, d.end)),
                            &sm.morph_influences,
                            None,
                        );
                    }
                }
                ObjectKind::LineSegments(ls) => {
                    push(&ls.geometry, &ls.material, primitive::LINE, None)
                }
                ObjectKind::Points(p) => push(&p.geometry, &p.material, primitive::POINT, None),
                ObjectKind::Sprite(s) => {
                    if let Some(g) = sprite_geometry(s) {
                        let (rotation, size_attenuation) = match &*s.material {
                            Material::Sprite(m) => (m.rotation, m.size_attenuation),
                            _ => (0.0, true),
                        };
                        // The billboard faces view 0; a pass per view
                        // promotes its own view to slot 0, so each faces its
                        // own camera. Whether the size follows depth is read
                        // off the primary view's projection.
                        let by_depth = !size_attenuation && primary_perspective;
                        let params = [
                            s.center.x - 0.5,
                            s.center.y - 0.5,
                            rotation,
                            if by_depth { 1.0 } else { 0.0 },
                        ];
                        push_skinned(
                            &g,
                            &s.material,
                            primitive::TRIANGLE,
                            None,
                            None,
                            Some(params),
                            g.draw_span(None),
                            &[],
                            None,
                        );
                    }
                }
                ObjectKind::InstancedMesh(im) => {
                    if im.transforms.is_empty() {
                        return;
                    }
                    let mut data = Vec::with_capacity(im.transforms.len() * 16);
                    for m in &im.transforms {
                        data.extend_from_slice(&m.elements);
                    }
                    // Per-instance morph influences (`set_morph_at`), when the
                    // mesh has them for every instance.
                    let row_len = im.morph_row_len();
                    let rows = (!im.morph_texture.is_empty()
                        && im.morph_texture.len() == row_len * im.transforms.len())
                    .then(|| im.morph_texture.clone());
                    if !im.geometry.attributes.contains_key("position") {
                        push(&im.geometry, &im.material, primitive::TRIANGLE, Some(data));
                        return;
                    }
                    // three.js `instanceColor`: white past the end of `colors`.
                    let colors = (!im.colors.is_empty()).then(|| {
                        (0..im.transforms.len())
                            .flat_map(|i| {
                                let c = im.colors.get(i).copied().unwrap_or(crate::math::Color::WHITE);
                                [c.r, c.g, c.b, 1.0]
                            })
                            .collect::<Vec<f32>>()
                    });
                    // One instanced item per draw — per group with a material
                    // array — as three.js's render list holds them.
                    for d in im.draws() {
                        *pending_colors.borrow_mut() = colors.clone();
                        push_skinned(
                            &im.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            Some(data.clone()),
                            None,
                            None,
                            Some((d.start, d.end)),
                            &im.morph_influences,
                            rows.clone().map(|r| (r, row_len)),
                        );
                    }
                }
                ObjectKind::Light(light) => match light {
                    Light::Ambient(l) => {
                        frame_u.ambient[0] += l.color.r * l.intensity;
                        frame_u.ambient[1] += l.color.g * l.intensity;
                        frame_u.ambient[2] += l.color.b * l.intensity;
                    }
                    Light::Directional(l) => {
                        if n_dir >= MAX_DIR_LIGHTS {
                            return;
                        }
                        // three.js: direction = normalize(target - position),
                        // default target the origin — matched to the wgpu path.
                        let pos = obj.world_position();
                        let d = Vector3::new(-pos.x, -pos.y, -pos.z);
                        let dir = if d.length_sq() > 1e-8 {
                            d.normalize()
                        } else {
                            transform_direction(&world, l.direction).normalize()
                        };
                        frame_u.dir[n_dir] = DirLightGpu {
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: scaled(l.color, l.intensity),
                        };
                        // Every casting light gets a record and layers of
                        // its kind's map array, and names the record.
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Dir { dir, settings: l.shadow });
                            frame_u.dir[n_dir].color[3] = casters.len() as f32;
                        }
                        n_dir += 1;
                    }
                    Light::Point(l) => {
                        if n_point >= MAX_POINT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        frame_u.points[n_point] = PointLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            color: scaled(l.color, l.intensity),
                            params: [l.distance, l.decay, 0.0, 0.0],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            // The cube's depth is normalised by the light's
                            // falloff `distance`, or — with none, as three.js's
                            // `PointLightShadow` does (`light.distance ||
                            // camera.far`) — by its shadow camera's far plane.
                            let radius = if l.distance > 0.0 { l.distance } else { l.shadow.camera_far };
                            casters.push(Caster::Point { pos: p, radius, settings: l.shadow });
                            frame_u.points[n_point].color[3] = casters.len() as f32;
                        }
                        n_point += 1;
                    }
                    Light::Spot(l) => {
                        if n_spot >= MAX_SPOT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let dir = transform_direction(&world, l.direction).normalize();
                        let casts = l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS;
                        if casts {
                            casters.push(Caster::Spot { pos: p, dir, angle: l.angle, settings: l.shadow });
                        }
                        frame_u.spots[n_spot] = SpotLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: scaled(l.color, l.intensity),
                            params: [
                                l.distance,
                                l.decay,
                                l.angle.cos(),
                                (l.angle * (1.0 - l.penumbra)).cos(),
                            ],
                        };
                        if casts {
                            frame_u.spots[n_spot].color[3] = casters.len() as f32;
                        }
                        n_spot += 1;
                    }
                    Light::Hemisphere(l) => {
                        if n_hemi >= MAX_HEMI_LIGHTS {
                            return;
                        }
                        let up = transform_direction(&world, Vector3::UP).normalize();
                        frame_u.hemis[n_hemi] = HemiLightGpu {
                            sky: scaled(l.sky_color, l.intensity),
                            ground: scaled(l.ground_color, l.intensity),
                            up: [up.x, up.y, up.z, 0.0],
                        };
                        n_hemi += 1;
                    }
                    // three.js orients it by its world matrix's rotation
                    // alone (`extractRotation`): width along local +x,
                    // height along +y, emitting toward local −z.
                    Light::RectArea(l) => {
                        let n = frame_u.rect_counts[0] as usize;
                        if n >= MAX_RECT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let ax = transform_direction(&world, Vector3::new(1.0, 0.0, 0.0)).normalize();
                        let ay = transform_direction(&world, Vector3::new(0.0, 1.0, 0.0)).normalize();
                        let (hw, hh) = (l.width * 0.5, l.height * 0.5);
                        frame_u.rects[n] = RectLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            color: scaled(l.color, l.intensity),
                            right: [ax.x * hw, ax.y * hw, ax.z * hw, 0.0],
                            up: [ay.x * hh, ay.y * hh, ay.z * hh, 0.0],
                        };
                        frame_u.rect_counts[0] += 1;
                    }
                },
                ObjectKind::Group => {}
            }
        });

        frame_u.counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        stats.geometry_bytes = self.geometries.values().map(|g| g.bytes).sum();
        // Logged on meaningful change rather than per frame: on a scene that
        // grows as it animates this is the number that predicts a stall, and
        // it is invisible otherwise.
        let last = self
            .logged_geometry_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        if stats.geometry_bytes.abs_diff(last) > (last / 8).max(64 << 20) {
            self.logged_geometry_bytes
                .store(stats.geometry_bytes, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "geometry buffers: {:.2} GB across {} cached ({})",
                stats.geometry_bytes as f64 / (1u64 << 30) as f64,
                self.geometries.len(),
                if slim_lines_enabled() { "slim lines" } else { "full vertices" },
            );
        }
        (frame_u, items, stats, casters)
    }

    // --------------------------------------------------------------- uploads

    /// Ensure `geometry` has current GPU buffers. `true` if it uploaded.
    fn ensure_geometry(
        &mut self,
        geometry: &Arc<BufferGeometry>,
        slim: bool,
    ) -> Result<bool, MetalError> {
        let key = Arc::as_ptr(geometry) as usize;
        let frame = self.frame;
        if let Some(cached) = self.geometries.get_mut(&key) {
            if cached.version == geometry.geometry_version && cached.slim == slim {
                cached.last_used = frame;
                return Ok(false);
            }
        }
        // Built as one or the other, never both: the slim path also skips
        // `derive_normals`, whose scratch vector is the size of the geometry.
        let (vbuf, vertex_count) = if slim {
            let vertices = build_line_vertices(geometry);
            if vertices.is_empty() {
                return Ok(false);
            }
            let n = vertices.len();
            (self.device.new_buffer(bytemuck::cast_slice(&vertices))?, n)
        } else {
            let vertices = build_vertices(geometry);
            if vertices.is_empty() {
                return Ok(false);
            }
            let n = vertices.len();
            (self.device.new_buffer(bytemuck::cast_slice(&vertices))?, n)
        };
        // Skinning attributes, if this geometry has them. Named as three.js
        // names them.
        let skin = if !slim
            && geometry.attributes.contains_key("joint")
            && geometry.attributes.contains_key("weight")
        {
            let data = build_skin(geometry, vertex_count);
            Some(self.device.new_buffer(bytemuck::cast_slice(&data))?)
        } else {
            None
        };
        let (ibuf, index_count) = match &geometry.index {
            Some(idx) if !idx.is_empty() => (
                Some(self.device.new_buffer(bytemuck::cast_slice(idx))?),
                idx.len(),
            ),
            _ => (None, 0),
        };
        // Morph targets in three.js's morph-texture layout, built by the
        // same helper the wgpu backend uploads from.
        let morph = match (!slim)
            .then(|| crate::core::MorphTexels::build(geometry))
            .flatten()
            .filter(|t| t.targets > 0 && t.vertex_count as usize == vertex_count)
        {
            Some(t) => {
                let has_position = geometry
                    .morph_attributes
                    .get("position")
                    .is_some_and(|v| !v.is_empty());
                Some(MorphBuffer {
                    buffer: self.device.new_buffer(bytemuck::cast_slice(&t.data))?,
                    count: t.targets,
                    stride: t.stride,
                    // Normal and colour as the helper says; position only
                    // with position targets, as `Morph` blends it; colour
                    // alpha (all four channels) for a 4-component colour.
                    flags: (t.flags & 3)
                        | (has_position as u32) << 2
                        | (((t.flags & 4) != 0) as u32) << 3,
                    vertex_count: t.vertex_count,
                })
            }
            None => None,
        };
        let morph_bytes = morph
            .as_ref()
            .map(|m| (m.count * m.stride * m.vertex_count) as u64 * 16)
            .unwrap_or(0);
        let sets = ["uv1", "uv2", "uv3"].map(|n| geometry.get_attribute(n).filter(|a| a.item_size >= 2));
        let tangents = geometry.get_attribute("tangent").filter(|a| a.item_size >= 3);
        let has_sets = sets.iter().any(Option::is_some);
        let uv_sets = if !slim && (has_sets || tangents.is_some()) {
            let mut data = vec![0.0f32; vertex_count * 12];
            for (set, attr) in sets.iter().enumerate() {
                let Some(a) = attr else { continue };
                for v in 0..vertex_count.min(a.count()) {
                    let at = v * 12 + (set / 2) * 4 + (set % 2) * 2;
                    data[at] = a.array[v * a.item_size];
                    data[at + 1] = a.array[v * a.item_size + 1];
                }
            }
            if let Some(t) = tangents {
                for v in 0..vertex_count.min(t.count()) {
                    let at = v * 12 + 8;
                    data[at..at + 3].copy_from_slice(&t.array[v * t.item_size..v * t.item_size + 3]);
                    data[at + 3] = if t.item_size >= 4 { t.array[v * t.item_size + 3] } else { 1.0 };
                }
            }
            let flags = has_sets as u32 | (tangents.is_some() as u32) << 1;
            Some((self.device.new_buffer(bytemuck::cast_slice(&data))?, flags))
        } else {
            None
        };
        self.geometries.insert(
            key,
            CachedGeometry {
                morph,
                uv_sets,
                bytes: morph_bytes + (vertex_count * if slim {
                    std::mem::size_of::<LineVertex>()
                } else {
                    std::mem::size_of::<Vertex>()
                } + index_count * std::mem::size_of::<u32>()) as u64,
                sphere: geometry_sphere(geometry),
                _geometry: geometry.clone(),
                version: geometry.geometry_version,
                vertices: vbuf,
                skin,
                indices: ibuf,
                index_count,
                vertex_count,
                wire: None,
                slim,
                last_used: frame,
            },
        );
        Ok(true)
    }

    /// Ensure the cached `geometry` has its wireframe line-list indices.
    ///
    /// Call after [`ensure_geometry`](Self::ensure_geometry): a stale entry
    /// is rebuilt there without them, so they follow `geometry_version`.
    fn ensure_wire_indices(&mut self, geometry: &Arc<BufferGeometry>) -> Result<(), MetalError> {
        let key = Arc::as_ptr(geometry) as usize;
        let Some(cached) = self.geometries.get_mut(&key) else {
            return Ok(());
        };
        if cached.wire.is_some() || cached.slim {
            return Ok(());
        }
        let flat = geometry.wireframe_indices();
        if flat.is_empty() {
            return Ok(());
        }
        let buf = self.device.new_buffer(bytemuck::cast_slice(&flat))?;
        cached.bytes += (flat.len() * std::mem::size_of::<u32>()) as u64;
        cached.wire = Some((buf, flat.len()));
        Ok(())
    }

    /// Upload `texture`, as colour or as data.
    ///
    /// `linear` asks for the bytes as they are, with no sRGB decode on sample.
    /// Normal, metallic-roughness and occlusion maps are *quantities*, not
    /// colours — the glTF spec says so in §3.9.2 — and decoding them through a
    /// transfer curve corrupts them: the 0.5 that means "flat" in a normal map
    /// comes out as 0.214, a tangent-space x of −0.57, so every flat texel
    /// arrives steeply tilted. The wgpu loader learned this the hard way and
    /// keeps two views of each image; this keeps two uploads, because a Metal
    /// texture's pixel format is fixed when it is created.
    ///
    /// The two are cached separately, so an image used as both a colour map and
    /// a data map gets one of each rather than whichever role asked first.
    fn ensure_texture_as(
        &mut self,
        texture: &Arc<Texture>,
        linear: bool,
    ) -> Result<bool, MetalError> {
        let key = texture_key(texture, linear);
        let sampler_key = SamplerKey::from_texture(texture);
        if !self.samplers.contains_key(&sampler_key) {
            let sampler = new_sampler(&self.device, sampler_key)?;
            self.samplers.insert(sampler_key, sampler);
        }
        let frame = self.frame;
        if let Some(cached) = self.textures.get_mut(&key) {
            cached.last_used = frame;
            return Ok(false);
        }
        // An unsupported format is not fatal: the material falls back to its
        // flat colour, which is a great deal more useful than no frame at all.
        let Ok(gpu) = upload_texture_as(&self.device, texture, linear) else {
            return Ok(false);
        };
        self.textures.insert(
            key,
            CachedTexture {
                _texture: texture.clone(),
                texture: gpu,
                sampler_key,
                last_used: frame,
            },
        );
        Ok(true)
    }

    /// The `MTLDepthStencilState` for one compare function and write flag,
    /// made on first use.
    fn depth_state(&mut self, compare: NSUInteger, write: bool) -> Result<Id, MetalError> {
        if !self.depth_states.contains_key(&(compare, write)) {
            let state = depth_stencil_state(&self.device, compare, write)?;
            self.depth_states.insert((compare, write), state);
        }
        Ok(self.depth_states[&(compare, write)].id())
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<Id, MetalError> {
        if !self.pipelines.contains_key(&key) {
            let state = render_pipeline(&self.device, key)?;
            self.pipelines.insert(key, state);
        }
        Ok(self.pipelines[&key].id())
    }

    // -------------------------------------------------------------- encoding

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode(
        &mut self,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
        vp: Option<(ClearMode, Option<PixelRect>)>,
    ) -> Result<(), MetalError> {
        let cmd = self.device.command_buffer();
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }

        // Order-independent transparency in passes of its own after each
        // view's — not in a layered (stereo) pass, where `Oit` draws sorted
        // with the other transparent surfaces, as `Glass` does. The depth it
        // tests against has to outlive the main pass for it.
        let layered_pass = vp.is_none() && layered(views, pass);
        self.oit_active = !layered_pass && !pass.depth.is_null() && items.iter().any(|i| i.oit && !i.capture_skip);
        let store_depth = self.store_depth;
        self.store_depth |= self.oit_active;
        let result = self.encode_views(cmd, pass, views, frame_u, items, scene, slot, stats, vp, layered_pass);
        self.store_depth = store_depth;
        self.oit_active = false;
        result?;

        if !pass.drawable.is_null() {
            let _: () = msg1(cmd, sel!("presentDrawable:"), pass.drawable);
        }
        let _: () = msg0(cmd, sel!("commit"));
        Ok(())
    }

    /// `encode`'s passes, one per view (or one layered pass for them all).
    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_views(
        &mut self,
        cmd: Id,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
        vp: Option<(ClearMode, Option<PixelRect>)>,
        layered_pass: bool,
    ) -> Result<(), MetalError> {
        if layered_pass {
            let descriptor =
                self.render_pass_descriptor(pass, scene, &views[0], true, false, views.len() as u32);
            self.encode_pass(
                cmd,
                descriptor,
                pass,
                frame_u,
                views[0].viewport,
                None,
                items,
                slot,
                stats,
                views.len() as u32,
                None,
                None,
            )?;
        } else {
            // One pass per view. The clear happens the first time a given
            // (texture, slice) is targeted; a second view sharing it loads
            // instead, or it would erase the first.
            let mut cleared: Vec<(Id, u32)> = Vec::with_capacity(views.len());
            for (index, view) in views.iter().enumerate() {
                let target = (
                    if view.color.is_null() {
                        pass.color
                    } else {
                        view.color
                    },
                    view.slice,
                );
                let first = !cleared.contains(&target);
                cleared.push(target);
                // Under the viewport state the first touch clears only when
                // the clear is the whole target; a scissored clear loads and
                // fills its rectangle, and auto-clear off just loads.
                let (clear, fill) = match vp {
                    None => (first, None),
                    Some((ClearMode::Full, _)) => (first, None),
                    Some((ClearMode::Rect(r), _)) if first => (
                        false,
                        Some((r, clear_color(scene.background, scene.background_alpha))),
                    ),
                    Some(_) => (false, None),
                };
                // A later view loads this target, so a multisampled colour has
                // to survive the resolve — `MULTISAMPLE_RESOLVE` alone discards
                // it and the next pane would load garbage.
                let keep = views[index + 1..].iter().any(|v| {
                    (if v.color.is_null() { pass.color } else { v.color }, v.slice) == target
                });

                // The mono shaders read view 0, so the view being drawn moves
                // into that slot for this pass.
                let mut per_view = *frame_u;
                per_view.promote_view(index);
                // The view now reads as view 0; its capture slice is `index`.
                per_view.rect_counts[3] = index as u32;
                // Pixel-sized effects (wide lines) convert through the viewport
                // they are drawn in, not the whole attachment.
                if let Some(v) = view.viewport {
                    let (w, h) = (v.width.max(1.0) as f32, v.height.max(1.0) as f32);
                    per_view.viewport = [w, h, 1.0 / w, 1.0 / h];
                }
                let descriptor =
                    // Kept multisampled when the OIT composite loads it next.
                    self.render_pass_descriptor(pass, scene, view, clear, keep || self.oit_active, 1);
                self.encode_pass(
                    cmd,
                    descriptor,
                    pass,
                    &per_view,
                    view.viewport,
                    Some(view.layers),
                    items,
                    slot,
                    stats,
                    1,
                    fill,
                    vp.and_then(|(_, s)| s),
                )?;
                if self.oit_active {
                    self.encode_oit(cmd, pass, view, &per_view, items, slot, stats, keep)?;
                }
            }
        }
        Ok(())
    }

    /// An item's world bounding sphere, where its geometry's bounds bound
    /// what it draws: not instanced, skinned, morphed or a sprite (each
    /// draws somewhere its object-space bounds do not say).
    fn item_sphere(&self, item: &DrawItem) -> Option<crate::math::Sphere> {
        if item.instance_count > 1
            || item.instance_slot.is_some()
            || item.bone_slot.is_some()
            || item.sprite.is_some()
            || item.morph_influences.iter().any(|&w| w != 0.0)
        {
            return None;
        }
        let g = self.geometries.get(&(Arc::as_ptr(&item.geometry) as usize))?;
        Some(g.sphere?.apply_matrix4(&item.world))
    }

    /// The OIT surfaces of one view: accumulated against the depth its main
    /// pass left — weighted light added up, transmittance multiplied down —
    /// then composited over its colour, multiplied by the transmittance and
    /// plus the light. Into the same (multisampled) colour the main pass drew
    /// and resolved again, so a later view's resolve keeps the composite.
    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_oit(
        &mut self,
        cmd: Id,
        pass: &PassAttachments,
        view: &RenderView,
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        slot: usize,
        stats: &mut MetalRenderStats,
        keep_multisampled: bool,
    ) -> Result<(), MetalError> {
        if !items.iter().any(|i| i.oit && !i.capture_skip && view.layers.test(&i.layers)) {
            return Ok(());
        }
        let samples = pass.sample_count.max(1);
        let (w, h) = (pass.width.max(1), pass.height.max(1));
        if self.oit_targets.as_ref().map(|t| t.size) != Some((w, h, samples)) {
            let usage = texture_usage::RENDER_TARGET | texture_usage::SHADER_READ;
            let mk = |n| {
                crate::metal::resources::new_texture_2d(
                    &self.device,
                    w,
                    h,
                    pixel_format::RGBA16_FLOAT,
                    usage,
                    storage_mode::PRIVATE,
                    n,
                    1,
                )
            };
            let (accum, reveal) = (mk(1)?, mk(1)?);
            let (accum_ms, reveal_ms) = if samples > 1 { (Some(mk(samples)?), Some(mk(samples)?)) } else { (None, None) };
            self.oit_targets = Some(OitTargets { size: (w, h, samples), accum, reveal, accum_ms, reveal_ms });
        }
        let key = (pass.color_format, samples);
        if !self.oit_composite.contains_key(&key) {
            let tint = crate::metal::resources::oit_composite_pipeline(&self.device, pass.color_format, samples, false)?;
            let light = crate::metal::resources::oit_composite_pipeline(&self.device, pass.color_format, samples, true)?;
            self.oit_composite.insert(key, (tint, light));
        }
        let t = self.oit_targets.as_ref().expect("made above");
        let (accum, reveal) = (t.accum.id(), t.reveal.id());
        let (accum_ms, reveal_ms) = (t.accum_ms.as_ref().map(|t| t.id()), t.reveal_ms.as_ref().map(|t| t.id()));

        // 1) Accumulation, against the main pass's depth.
        let rpd: Id = msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor"));
        let attachments: Id = msg0(rpd, sel!("colorAttachments"));
        for (index, (single, multi, clear)) in
            [(accum, accum_ms, MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 }),
             (reveal, reveal_ms, MTLClearColor { red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0 })]
            .into_iter()
            .enumerate()
        {
            let a: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), index);
            let _: () = msg1(a, sel!("setLoadAction:"), load_action::CLEAR);
            let _: () = msg1(a, sel!("setClearColor:"), clear);
            match multi {
                Some(m) => {
                    let _: () = msg1(a, sel!("setTexture:"), m);
                    let _: () = msg1(a, sel!("setResolveTexture:"), single);
                    let _: () = msg1(a, sel!("setStoreAction:"), store_action::MULTISAMPLE_RESOLVE);
                }
                None => {
                    let _: () = msg1(a, sel!("setTexture:"), single);
                    let _: () = msg1(a, sel!("setStoreAction:"), store_action::STORE);
                }
            }
        }
        let depth_texture = if view.depth.is_null() { pass.depth } else { view.depth };
        let depth: Id = msg0(rpd, sel!("depthAttachment"));
        let _: () = msg1(depth, sel!("setTexture:"), depth_texture);
        let _: () = msg1(depth, sel!("setSlice:"), view.slice as NSUInteger);
        let _: () = msg1(depth, sel!("setLoadAction:"), load_action::LOAD);
        let _: () = msg1(depth, sel!("setStoreAction:"), store_action::DONT_CARE);
        self.oit_phase = 1;
        let accumulated = self.encode_pass(cmd, rpd, pass, frame_u, view.viewport, Some(view.layers), items, slot, stats, 1, None, None);
        self.oit_phase = 0;
        accumulated?;

        // 2) The composite, over the colour the main pass drew.
        let rpd: Id = msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor"));
        let attachments: Id = msg0(rpd, sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let color_texture = if view.color.is_null() { pass.color } else { view.color };
        let _: () = msg1(color, sel!("setTexture:"), color_texture);
        let _: () = msg1(color, sel!("setSlice:"), view.slice as NSUInteger);
        let _: () = msg1(color, sel!("setLoadAction:"), load_action::LOAD);
        if pass.resolve.is_null() {
            let _: () = msg1(color, sel!("setStoreAction:"), store_action::STORE);
        } else {
            let _: () = msg1(color, sel!("setResolveTexture:"), pass.resolve);
            let _: () = msg1(color, sel!("setResolveSlice:"), view.slice as NSUInteger);
            let _: () = msg1(
                color,
                sel!("setStoreAction:"),
                if keep_multisampled { store_action::STORE_AND_MULTISAMPLE_RESOLVE } else { store_action::MULTISAMPLE_RESOLVE },
            );
        }
        let encoder: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), rpd);
        if encoder.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        if let Some(v) = view.viewport {
            let _: () = msg1(encoder, sel!("setViewport:"), v);
        }
        let (tint, light) = &self.oit_composite[&key];
        let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), accum, 0usize);
        let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), reveal, 1usize);
        for pipe in [tint.id(), light.id()] {
            let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipe);
            let _: () = msg3(encoder, sel!("drawPrimitives:vertexStart:vertexCount:"), primitive::TRIANGLE, 0usize, 3usize);
        }
        let _: () = msg0(encoder, sel!("endEncoding"));
        Ok(())
    }

    /// Encode every item into one render pass.
    ///
    /// `views_per_draw` is 1 for an ordinary pass and the view count for a
    /// layered one, where the instance count is multiplied by it and the vertex
    /// stage splits the id back into (view, instance).
    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_pass(
        &mut self,
        cmd: Id,
        descriptor: Id,
        pass: &PassAttachments,
        frame_u: &FrameUniforms,
        viewport: Option<MTLViewport>,
        view_layers: Option<Layers>,
        items: &[DrawItem],
        slot: usize,
        stats: &mut MetalRenderStats,
        views_per_draw: u32,
        fill: Option<(PixelRect, MTLClearColor)>,
        scissor_test: Option<PixelRect>,
    ) -> Result<(), MetalError> {
        // Built before the encoder exists: an error after it would have to
        // end the encoder first.
        let fill_pipeline = match fill {
            Some(_) => Some(self.viewport_state.fill_pipeline(
                pass.color_format,
                pass.depth_format,
                pass.sample_count.max(1),
            )?),
            None => None,
        };
        let encoder: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), descriptor);
        if encoder.is_null() {
            return Err(MetalError::Unsupported(
                "render command encoder could not be created for these attachments".into(),
            ));
        }

        let _: () = msg1(
            encoder,
            sel!("setFrontFacingWinding:"),
            winding::COUNTER_CLOCKWISE,
        );
        // The scissored clear, before anything draws: the background over the
        // scissor rectangle and nowhere else.
        if let (Some((rect, color)), Some(pipeline)) = (fill, fill_pipeline) {
            super::viewport::encode_fill(encoder, pipeline, rect, color);
        }
        // With the scissor test on, nothing lands outside its rectangle.
        let mut items = items;
        if let Some(viewport) = viewport {
            let _: () = msg1(encoder, sel!("setViewport:"), viewport);
            // Scissor to the viewport, clamped to the attachment (Metal
            // rejects a scissor outside it), so nothing a pane draws — a wide
            // line, a point sprite — can land in the pane next to it.
            use super::viewport::{intersect, scissor_for_viewport, scissor_rect};
            let rect = scissor_for_viewport(&viewport, pass.width, pass.height).and_then(|r| {
                match scissor_test {
                    Some(s) => intersect(r, scissor_rect(s)),
                    None => Some(r),
                }
            });
            match rect {
                Some(rect) => {
                    let _: () = msg1(encoder, sel!("setScissorRect:"), rect);
                }
                // Viewport and scissor miss each other: nothing to draw.
                None => items = &[],
            }
        }
        // The frame uniforms outgrow `setVertexBytes`'s 4 KB with the shadow
        // records, so they go in a buffer of this frame's, kept until the
        // frame in flight is done with it.
        let frame_bytes = std::slice::from_raw_parts(
            frame_u as *const FrameUniforms as *const u8,
            std::mem::size_of::<FrameUniforms>(),
        );
        let frame_buf = match self.device.new_buffer(frame_bytes) {
            Ok(b) => b,
            Err(e) => {
                let _: () = msg0(encoder, sel!("endEncoding"));
                return Err(e);
            }
        };
        let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), frame_buf.id(), 0usize, VB_FRAME);
        let _: () = msg3(encoder, sel!("setFragmentBuffer:offset:atIndex:"), frame_buf.id(), 0usize, FB_FRAME);
        self.transients[slot].push(frame_buf);

        let mut last_pipeline = NIL;
        let mut last_cull = usize::MAX;
        let mut last_fill = usize::MAX;
        let mut last_depth = NIL;
        let mut last_bias = (0.0f32, 0.0f32);
        // A failure inside the loop still has to close the encoder: Metal
        // aborts the process if one is released without `endEncoding`.
        let mut failure = None;

        // Frustum culling, against this view's view-projection: an item whose
        // bounds are outside it is not encoded at all. Not in a layered pass,
        // where one draw serves both eyes and either could see it.
        let cull = (views_per_draw == 1)
            .then(|| crate::math::Frustum::from_projection_matrix(&Matrix4 { elements: frame_u.view_proj[0] }));
        for item in items {
            if let Some(f) = &cull {
                if self.item_sphere(item).is_some_and(|s| !f.intersects_sphere(&s)) {
                    stats.culled += 1;
                    continue;
                }
            }
            // With an OIT pass this frame, its items draw there and nowhere
            // else.
            if self.oit_active && item.oit != (self.oit_phase == 1) {
                continue;
            }
            if item.capture_skip {
                continue;
            }
            if let Some(layers) = view_layers {
                if !layers.test(&item.layers) {
                    continue;
                }
            }
            let key = Arc::as_ptr(&item.geometry) as usize;
            // Nothing uploaded for this geometry — an attribute set the vertex
            // builder could make nothing of. Counted, not drawn, not fatal.
            let Some(geom) = self.geometries.get(&key) else {
                stats.skipped += 1;
                continue;
            };
            // Everything needed from the cache entry, read out before anything
            // borrows `self` mutably again.
            let (vertices, indices, index_count, vertex_count, skin_buffer) = (
                geom.vertices.id(),
                geom.indices.as_ref().map(|b| b.id()),
                geom.index_count,
                geom.vertex_count,
                geom.skin.as_ref().map(|b| b.id()),
            );
            let morph_u = item_morph_uniforms(item, geom.morph.as_ref());
            let morph_buffer = geom
                .morph
                .as_ref()
                .filter(|_| morph_u.info[0] > 0)
                .map(|m| m.buffer.id())
                .unwrap_or(self.identity_instance.id());
            let uv_sets = geom.uv_sets.as_ref().map(|(b, f)| (b.id(), *f));
            // A wireframe mesh draws its line-list (polygon outlines, 1px
            // hardware lines, as the wgpu `TriangleWire` path does) rather
            // than its triangles in line fill mode, which would show every
            // fan diagonal. Fill-mode LINES stays only as the fallback for a
            // wireframe with no line list (nothing to outline). A partial
            // span draws the line list built for it.
            let wire = if item.primitive == primitive::TRIANGLE && item.material.wireframe() {
                match item.wire_span {
                    Some((i, n)) => self.transients[slot].get(i).map(|b| (b.id(), n)),
                    None if item.range == (0, item.geometry.draw_count() as u32) => {
                        geom.wire.as_ref().map(|(b, n)| (b.id(), *n))
                    }
                    None => None,
                }
            } else {
                None
            };
            let draw_primitive = if wire.is_some() {
                primitive::LINE
            } else {
                item.primitive
            };

            let material = &item.material;
            let point_sprites = item.primitive == primitive::POINT;
            let layered_draw = views_per_draw > 1;
            let wide = item.primitive == primitive::LINE
                && self.line_width > 0.0
                && !layered_draw
                && slim_lines_enabled();
            // Both halves have to be true: a mesh with a skeleton whose
            // geometry carries no weights has nothing to blend, and a geometry
            // with weights and no skeleton has nothing to blend *through*.
            // Read before `self.pipeline`, which borrows `self` mutably.
            let is_skinned = item.bone_slot.is_some() && skin_buffer.is_some() && !layered_draw;
            let fur_draw = item.fur.is_some() && !layered_draw && !point_sprites && item.primitive == primitive::TRIANGLE;
            // An `fx` surface draws in screen space from its own uniforms; a
            // layered (stereo) pass has no single viewport for it, so it is
            // skipped there.
            if item.fx_surface.is_some() && (layered_draw || item.primitive != primitive::TRIANGLE) {
                continue;
            }
            let fx_draw = item.fx_surface.is_some();
            let pipeline = match self.pipeline(PipelineKey {
                point_sprites,
                fur: fur_draw,
                fx_surface: fx_draw,
                wide_lines: wide,
                // Must agree with what `ensure_geometry` uploaded, or the
                // vertex shader reads the wrong stride and the draw is garbage.
                slim_lines: item.primitive == primitive::LINE
                    && !layered_draw
                    && slim_lines_enabled(),
                blend: item.transparent,
                layered: layered_draw,
                // Both halves have to be true: a mesh with a skeleton whose
                // geometry carries no weights has nothing to blend, and a
                // geometry with weights and no skeleton has nothing to blend
                // *through*.
                skinned: is_skinned,
                // Only the layered path needs this, and naming it there costs a
                // pipeline per topology — so leave it unspecified elsewhere.
                topology: if layered_draw {
                    topology_class::of(draw_primitive)
                } else {
                    topology_class::UNSPECIFIED
                },
                color_format: pass.color_format,
                depth_format: pass.depth_format,
                sample_count: pass.sample_count.max(1),
                color_write: item.state.color_write,
                oit_accum: self.oit_phase == 1,
            }) {
                Ok(pipeline) => pipeline,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            if pipeline != last_pipeline {
                let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipeline);
                last_pipeline = pipeline;
            }

            // The material's `depthTest` / `depthWrite` / `depthFunc`. By
            // default transparent surfaces test depth but do not write it, so
            // two blended layers both survive rather than the nearer one
            // erasing the further one's contribution; with the test off
            // nothing is written, as in WebGL. A pass with no depth attachment
            // gets no depth state at all: Metal rejects one that would write.
            if !pass.depth.is_null() {
                let mut func = item.state.effective_func();
                if pass.reverse_z {
                    func = func.reversed();
                }
                // The accumulation reads the opaque depth and writes none.
                let write = item.state.effective_write(!item.transparent) && self.oit_phase == 0;
                let depth_state = match self.depth_state(metal_compare(func), write) {
                    Ok(s) => s,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                if depth_state != last_depth {
                    let _: () = msg1(encoder, sel!("setDepthStencilState:"), depth_state);
                    last_depth = depth_state;
                }
                // three.js `polygonOffset`: Metal's depth bias is
                // `glPolygonOffset`'s formula — `slopeScale · m + bias · r` —
                // so factor and units carry over, negated under reverse-Z
                // where nearer is larger.
                let sign = if pass.reverse_z { -1.0 } else { 1.0 };
                let bias = material
                    .polygon_offset()
                    .map(|o| (o.units * sign, o.factor * sign))
                    .unwrap_or((0.0, 0.0));
                if bias != last_bias {
                    let _: () = msg3(
                        encoder,
                        sel!("setDepthBias:slopeScale:clamp:"),
                        bias.0,
                        bias.1,
                        0.0f32,
                    );
                    last_bias = bias;
                }
            }

            let cull = match item.side_override.unwrap_or(material.side()) {
                1 => cull::FRONT,
                2 => cull::NONE,
                _ => cull::BACK,
            };
            if cull != last_cull {
                let _: () = msg1(encoder, sel!("setCullMode:"), cull);
                last_cull = cull;
            }
            let fill = if material.wireframe() && wire.is_none() {
                fill_mode::LINES
            } else {
                fill_mode::FILL
            };
            if fill != last_fill {
                let _: () = msg1(encoder, sel!("setTriangleFillMode:"), fill);
                last_fill = fill;
            }

            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                vertices,
                0usize,
                VB_VERTICES,
            );

            // Instanced draws read their matrices from this frame's transient
            // buffer; everything else still binds the 1x1 identity, because the
            // shader declares the buffer and Metal requires bound what is
            // declared.
            let instanced = item
                .instance_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id());
            let instance_buffer = instanced.unwrap_or(self.identity_instance.id());
            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                instance_buffer,
                0usize,
                VB_INSTANCES,
            );
            // Metal requires every resource a function declares to be bound,
            // and `vs_skinned` declares both — so bind them whenever they
            // exist, and fall back to the vertex buffer itself when they do
            // not. The non-skinned vertex functions never read either index.
            let skin_buffer = skin_buffer.unwrap_or(vertices);
            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                skin_buffer,
                0usize,
                VB_SKIN,
            );
            let bone_buffer = item
                .bone_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                bone_buffer,
                0usize,
                VB_BONES,
            );
            // Morph targets: the geometry's buffer, or a stand-in with count
            // 0 — the mesh functions declare both.
            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                morph_buffer,
                0usize,
                VB_MORPH_DATA,
            );
            set_bytes(encoder, true, &morph_u, VB_MORPH);
            // The extra uv sets, or a stand-in the shader never reads
            // (`phys_flags.z` is 0 without them).
            let _: () = msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                uv_sets.map_or(self.identity_instance.id(), |(b, _)| b),
                0usize,
                VB_UV_SETS,
            );
            if let (true, Some(fur)) = (fur_draw, &item.fur) {
                set_bytes(encoder, true, fur, VB_FUR);
                set_bytes(encoder, false, fur, FB_FUR);
            }
            // The surface reads its block at the fur's slots: the two never
            // share a draw.
            if let Some(fx) = &item.fx_surface {
                set_bytes(encoder, true, &**fx, VB_FUR);
                set_bytes(encoder, false, &**fx, FB_FUR);
            }

            let (texture, sampler) = self.material_texture(material);
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                texture,
                0usize,
            );
            // three.js gives every texture its own filter and wrap: each map
            // takes its sampler's place in the pool, the base map's first.
            let mut pool = SamplerPool::new(sampler);
            let (extra, present) = self.material_maps(material);
            for (i, slot) in extra.iter().enumerate() {
                let _: () = msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    i + 1,
                );
                // Emissive, normal, metallic-roughness, ambient occlusion.
                pool.set(1 + i, slot.map_or(sampler, |(_, s)| s));
            }
            // A Standard/Physical material's own `envMap` in place of the
            // scene's, with its own intensity and rotation.
            let own_env = material.pbr_env().and_then(|(c, intensity, rotation)| {
                let c = self.prefiltered_cached(c);
                self.material_envs
                    .get(&crate::renderer::gpu_texture::cube_cache_key(&c))
                    .map(|e| {
                        (e, intensity, crate::materials::environment_lookup_rows(rotation, e.atlas.is_none()))
                    })
            });
            let pbr_env = own_env.map(|e| e.0).or(self.env_cube.as_ref().map(|(_, e)| e));
            let env = pbr_env.map(|e| e.cube.id());
            // Its PMREM atlas, which `ibl` reads (texture 15, sampler 13).
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                pbr_env.and_then(|e| e.atlas.as_ref()).map_or(self.white.id(), |a| a.id()),
                15usize,
            );
            let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), self.ltc.id(), 28usize);
            // The transmission capture (three.js `transmissionSamplerMap`).
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                self.capture_bound.unwrap_or(self.capture_dummy.id()),
                29usize,
            );
            let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), self.transmission_sampler.id(), 14usize);
            let atlas_sampler = self
                .samplers
                .get(&SamplerKey::default_linear())
                .map_or(sampler, |s| s.id());
            let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), atlas_sampler, 13usize);
            let env_atlas_params = pbr_env.map_or([0.0; 4], |e| e.params);
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                env.unwrap_or(self.black_cube.id()),
                5usize,
            );
            // A mipmapped sampler for the cube: the environment is sampled by
            // roughness across its own chain, and a sampler that cannot see
            // past level 0 makes every surface a mirror.
            let env_sampler = self
                .samplers
                .get(&SamplerKey::environment())
                .map(|s| s.id())
                .unwrap_or(sampler);
            let _: () = msg2(
                encoder,
                sel!("setFragmentSamplerState:atIndex:"),
                env_sampler,
                10usize,
            );
            // The shadow map and its comparison sampler. Bound even when
            // nothing casts, because Metal requires every declared resource to
            // be bound and `shadow_factor` returns early on the flag anyway.
            let shadow_tex = self
                .shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                shadow_tex,
                6usize,
            );
            let shadow_samp = self
                .shadow_sampler
                .as_ref()
                .map(|s| s.id())
                .unwrap_or(env_sampler);
            let _: () = msg2(
                encoder,
                sel!("setFragmentSamplerState:atIndex:"),
                shadow_samp,
                11usize,
            );
            let spot_tex = self
                .spot_shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), spot_tex, 7usize);
            let point_tex = self
                .point_shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), point_tex, 8usize);
            // The variant-only maps: alpha, light, bump, specular at textures
            // 9–12, their samplers in the pool.
            let slots = material.texture_slots();
            // The light map is colour, the rest data.
            let extras = [
                (&slots.alpha_map, true),
                (&slots.light_map, false),
                (&slots.bump_map, true),
                (&slots.specular_map, true),
            ]
            .map(|(m, linear)| self.slot_texture(m.as_ref(), linear));
            for (i, slot) in extras.iter().enumerate() {
                let _: () = msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    9 + i,
                );
                pool.set(5 + i, slot.map_or(sampler, |(_, s)| s));
            }
            // MeshPhysicalMaterial's maps at textures 16–27.
            let mut phys_bits = 0u32;
            for (i, (map, linear)) in physical_maps(&slots).into_iter().enumerate() {
                let slot = self.slot_texture(map, linear);
                let _: () = msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    PHYS_TEXTURE0 + i,
                );
                if let Some((_, smp)) = slot {
                    pool.set(MAP_PHYS0 + i, smp);
                    phys_bits |= 1 << i;
                }
            }

            let mut draw_u = draw_uniforms(
                item,
                texture != self.white.id(),
                instanced.is_some(),
                &item.geometry,
            );
            draw_u.maps = present;
            draw_u.extra = extras.map(|e| e.is_some() as u32);
            // The material's own `envMap`, mipmapped like the scene's.
            let mat_env = slots
                .env_map
                .as_ref()
                .and_then(|c| self.material_envs.get(&crate::renderer::gpu_texture::cube_cache_key(c)))
                .map(|e| e.cube.id());
            draw_u.map_scalars[2] = mat_env.is_some() as u32 as f32;
            // A matcap's texture, sRGB like a colour map — or, on a toon
            // material, its `gradientMap`, decoded by its own colour space as
            // three.js does. No material has both.
            let matcap = self.slot_texture(slots.matcap_map.as_ref().or(slots.gradient_map.as_ref()), false);
            draw_u.map_scalars[3] = matcap.is_some() as u32 as f32;
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                matcap.map_or(self.white.id(), |(t, _)| t),
                14usize,
            );
            pool.set(9, matcap.map_or(sampler, |(_, s)| s));
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                mat_env.unwrap_or(self.black_cube.id()),
                13usize,
            );
            let mat_env_sampler = slots
                .env_map
                .as_ref()
                .and_then(|c| self.samplers.get(&SamplerKey::from_cube(c)))
                .map_or(env_sampler, |s| s.id());
            let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), mat_env_sampler, 12usize);
            for (i, smp) in pool.slots().into_iter().enumerate() {
                let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), smp, i);
            }
            draw_u.map_samplers = pool.packed();
            draw_u.phys_flags = [
                phys_bits,
                self.specular_antialiasing as u32,
                uv_sets.map_or(0, |(_, f)| f),
                item.blended_glass as u32,
            ];
            // three.js `envMapIntensity` with the material's own environment,
            // `scene.environmentIntensity` with the scene's.
            let (intensity, rows) = own_env.map_or(self.env_light, |e| (e.1, e.2));
            draw_u.map_params[2] = intensity;
            draw_u.map_params[3] = if env.is_some() { 1.0 } else { 0.0 };
            if matches!(material.kind(), MaterialKind::Standard | MaterialKind::Physical) {
                draw_u.env_dir = rows;
                draw_u.env_params = env_atlas_params;
            }
            if wide {
                // misc.y is the point size, which a line never uses — the quad
                // path reads its width in pixels from the same slot.
                draw_u.misc[1] = self.line_width;
                // misc.x, otherwise unused: whether the quad blends, and so
                // can draw coverage rather than cut at half.
                draw_u.misc[0] = if item.transparent { 1.0 } else { 0.0 };
            }
            set_bytes(encoder, true, &draw_u, VB_DRAW);
            set_bytes(encoder, false, &draw_u, FB_DRAW);

            // Without a matrix buffer there is exactly one instance to draw,
            // whatever the object claimed — the alternative is the shader
            // indexing past the identity matrix.
            let instances = if instanced.is_some() {
                item.instance_count.max(1)
            } else {
                1
            };
            // A layered pass draws every instance once per view; the vertex
            // stage divides the id back out.
            let instances = instances * views_per_draw as usize;
            // The item's span: three.js's draw range cut to its group. The
            // wireframe line list replaces the triangle indices for the draw
            // only (it was built for the span, or is the whole list for a
            // whole span); the triangle count below reports the span.
            let (start, end) = (item.range.0 as usize, item.range.1 as usize);
            let (draw_indices, draw_first, draw_index_count) = match wire {
                Some((buf, n)) => (Some(buf), 0usize, n),
                None => (indices, start, end.min(index_count).saturating_sub(start)),
            };
            match draw_indices {
                Some(ibuf) if index_count > 0 || wire.is_some() => {
                    if draw_index_count > 0 {
                        let _: () = msg6(
                            encoder,
                            sel!(
                                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                            ),
                            draw_primitive,
                            draw_index_count as NSUInteger,
                            index_type::UINT32,
                            ibuf,
                            draw_first * std::mem::size_of::<u32>(),
                            instances as NSUInteger,
                        );
                    }
                }
                _ => {
                    let end = end.min(vertex_count);
                    // Six vertices per segment instead of two, as triangles —
                    // the vertex buffer is unchanged, only how it is read.
                    let (prim, first, count) = if wide {
                        (primitive::TRIANGLE, (start / 2) * 6, (end.saturating_sub(start) / 2) * 6)
                    } else {
                        (item.primitive, start, end.saturating_sub(start))
                    };
                    if count > 0 {
                        let _: () = msg4(
                            encoder,
                            sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                            prim,
                            first,
                            count as NSUInteger,
                            instances as NSUInteger,
                        );
                    }
                }
            }

            stats.draw_calls += 1;
            if item.primitive == primitive::TRIANGLE {
                let verts = end.min(if index_count > 0 { index_count } else { vertex_count })
                    .saturating_sub(start);
                stats.triangles += (verts / 3 * instances) as u32;
            }
        }

        let _: () = msg0(encoder, sel!("endEncoding"));
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The texture + sampler to bind for `material`, falling back to the 1x1
    /// white texture so the shader can sample unconditionally.
    fn material_texture(&self, material: &Material) -> (Id, Id) {
        let slots = material.texture_slots();
        // A matcap's own texture has its own slot (`fs_mesh`'s `matcap_tex`).
        let map = slots.map.as_ref();
        if let Some(map) = map {
            if let Some(cached) = self.textures.get(&(Arc::as_ptr(map) as usize)) {
                if let Some(sampler) = self.samplers.get(&cached.sampler_key) {
                    return (cached.texture.id(), sampler.id());
                }
            }
        }
        // No colour map: the white stand-in, sampled with the first map
        // present's settings (`MaterialTextureSlots::sampler_source`), which
        // the other slots share.
        let from_maps = slots
            .sampler_source()
            // Cached under its role: sRGB for colour slots, linear for data.
            .and_then(|t| {
                self.textures
                    .get(&texture_key(t, false))
                    .or_else(|| self.textures.get(&texture_key(t, true)))
            })
            .and_then(|cached| self.samplers.get(&cached.sampler_key));
        let sampler = from_maps
            .or_else(|| self.samplers.get(&SamplerKey::default_linear()))
            .map(|s| s.id())
            .unwrap_or(NIL);
        (self.white.id(), sampler)
    }

    /// One uploaded texture and its sampler, or `None` if it has not been
    /// uploaded yet.
    ///
    /// Separate from [`material_texture`](Self::material_texture) because the
    /// extra slots have no fallback worth binding: a shader that is told the
    /// slot is empty never samples it, so binding white there would only hide
    /// an upload that did not happen.
    fn slot_texture(&self, map: Option<&Arc<Texture>>, linear: bool) -> Option<(Id, Id)> {
        let map = map?;
        let cached = self.textures.get(&texture_key(map, linear))?;
        let sampler = self.samplers.get(&cached.sampler_key)?;
        Some((cached.texture.id(), sampler.id()))
    }

    /// Draw every opaque triangle mesh into the shadow map, from the light.
    ///
    /// Transparent items are skipped: a depth buffer cannot represent partial
    /// occlusion, so a glass panel would cast the shadow of a solid one. That
    /// is three.js's behaviour too, and the alternative — a second pass with
    /// alpha testing — is a lot of machinery for a case that usually wants no
    /// shadow at all.
    unsafe fn encode_shadow(
        &mut self,
        pass_u: &ShadowPassGpu,
        items: &[DrawItem],
        slot: usize,
        map: Id,
        slice: usize,
        radial: bool,
        frustum: &crate::math::Frustum,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let chosen = if radial {
            self.point_shadow_pipeline.as_ref()
        } else {
            self.shadow_pipeline.as_ref()
        };
        let Some(pipeline) = chosen else {
            return Ok(());
        };
        let pipeline = pipeline.id();
        let cmd = unsafe { self.device.command_buffer() };
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let desc: Id = unsafe { msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor")) };
        unsafe {
            let depth: Id = msg0(desc, sel!("depthAttachment"));
            let _: () = msg1(depth, sel!("setTexture:"), map);
            let _: () = msg1(depth, sel!("setLoadAction:"), load_action::CLEAR);
            let _: () = msg1(depth, sel!("setStoreAction:"), store_action::STORE);
            let _: () = msg1(depth, sel!("setClearDepth:"), 1.0f64);
            // Which cube face, for a point light. Zero for the flat maps.
            let _: () = msg1(depth, sel!("setSlice:"), slice as NSUInteger);
        }
        let encoder: Id = unsafe { msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc) };
        if encoder.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let cut_pipeline = self.cut_shadow_pipeline.as_ref().map(|p| p.id());
        let mut current = pipeline;
        let mut last_cull = NSUInteger::MAX;
        unsafe {
            let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipeline);
            // Counter-clockwise front faces, as the colour pass draws them, so
            // a caster's `side` can choose which faces its shadow records.
            let _: () = msg1(encoder, sel!("setFrontFacingWinding:"), winding::COUNTER_CLOCKWISE);
            // Forward-Z `Less`, write on: what a shadow map wants (made in
            // `with_device`, so the lookup cannot miss).
            let _: () = msg1(
                encoder,
                sel!("setDepthStencilState:"),
                self.depth_states[&(compare::LESS, true)].id(),
            );
            set_bytes(encoder, true, pass_u, VB_FRAME);
            if radial {
                // `fs_shadow_point` reads the light's position and radius from
                // the same uniform, and the fragment stage has its own binding
                // table — binding it for the vertex stage alone leaves the
                // fragment reading zeros, which writes a depth of 0 everywhere
                // and shadows nothing.
                set_bytes(encoder, false, pass_u, VB_FRAME);
            }
        }

        for item in items {
            // Per-object, matching the wgpu path: `Object3D::cast_shadow` is
            // off by default and a caster has to say so. Shadowing everything
            // opaque instead looks right in a test scene and wrong in a real
            // one, where the floor casts onto itself.
            // A transparent caster casts too: three.js's shadow map ignores
            // blending (only `alphaTest` cuts a shadow).
            if !item.cast_shadow {
                continue;
            }
            if item.primitive != primitive::TRIANGLE {
                continue;
            }
            if self.item_sphere(item).is_some_and(|s| !frustum.intersects_sphere(&s)) {
                continue;
            }
            let key = Arc::as_ptr(&item.geometry) as usize;
            let Some(geom) = self.geometries.get(&key) else {
                continue;
            };
            let (vertices, indices, index_count, vertex_count, skin) = (
                geom.vertices.id(),
                geom.indices.as_ref().map(|b| b.id()),
                geom.index_count,
                geom.vertex_count,
                geom.skin.as_ref().map(|b| b.id()),
            );
            let instances = item
                .instance_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let bones = item
                .bone_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let morph_u = item_morph_uniforms(item, geom.morph.as_ref());
            let morph_buffer = geom
                .morph
                .as_ref()
                .filter(|_| morph_u.info[0] > 0)
                .map(|m| m.buffer.id())
                .unwrap_or(self.identity_instance.id());
            let (start, end) = (item.range.0 as usize, item.range.1 as usize);
            // The caster's map, for a cut-out test: `has_map` as the colour
            // pass computes it (anything but the white stand-in).
            let (map_texture, map_sampler) = self.material_texture(&item.material);
            let mut draw_u = draw_uniforms(
                item,
                map_texture != self.white.id(),
                item.instance_slot.is_some(),
                &item.geometry,
            );
            // `maps.w == 2` is how `vs_shadow` is told to skin; the fragment
            // stage never sees this uniform, so the slot is free.
            if item.bone_slot.is_some() && skin.is_some() {
                draw_u.maps[3] = 2;
            }
            // three.js's shadow depth material takes the caster's `map`,
            // `alphaMap` and `alphaTest`: a cut-out caster runs
            // `fs_shadow_cut` (the point pass's fragment stage tests it
            // already).
            let alpha = self.slot_texture(item.material.texture_slots().alpha_map.as_ref(), true);
            draw_u.extra[0] = alpha.is_some() as u32;
            let cut = item.material.alpha_test() > 0.0 && (draw_u.flags[1] == 1 || alpha.is_some());
            let want = match (radial, cut, cut_pipeline) {
                (false, true, Some(p)) => p,
                _ => pipeline,
            };
            // three.js's `shadowSide`: a front-sided caster records its back
            // faces, a back-sided one its front faces, a double-sided one both.
            let cull = shadow_cull(item.material.side(), radial);
            unsafe {
                if want != current {
                    let _: () = msg1(encoder, sel!("setRenderPipelineState:"), want);
                    current = want;
                }
                if cull != last_cull {
                    let _: () = msg1(encoder, sel!("setCullMode:"), cull);
                    last_cull = cull;
                }
                if cut || radial {
                    let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), map_texture, 0usize);
                    let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), map_sampler, 0usize);
                    let (alpha_tex, alpha_smp) = alpha.unwrap_or((self.white.id(), map_sampler));
                    let _: () = msg2(encoder, sel!("setFragmentTexture:atIndex:"), alpha_tex, 1usize);
                    let _: () = msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), alpha_smp, 1usize);
                    set_bytes(encoder, false, &draw_u, VB_DRAW);
                }
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), vertices, 0usize, 0usize);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), instances, 0usize, VB_INSTANCES);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), skin.unwrap_or(vertices), 0usize, VB_SKIN);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), bones, 0usize, VB_BONES);
                let _: () = msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), morph_buffer, 0usize, VB_MORPH_DATA);
                set_bytes(encoder, true, &morph_u, VB_MORPH);
                set_bytes(encoder, true, &draw_u, VB_DRAW);
                let n = item.instance_count.max(1) as NSUInteger;
                match indices {
                    Some(ibuf) if index_count > 0 => {
                        let count = end.min(index_count).saturating_sub(start);
                        if count > 0 {
                            let _: () = msg6(
                                encoder,
                                sel!(
                                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                                ),
                                primitive::TRIANGLE,
                                count as NSUInteger,
                                index_type::UINT32,
                                ibuf,
                                start * std::mem::size_of::<u32>(),
                                n,
                            );
                        }
                    }
                    _ => {
                        let count = end.min(vertex_count).saturating_sub(start);
                        if count > 0 {
                            let _: () = msg4(
                                encoder,
                                sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                                primitive::TRIANGLE,
                                start,
                                count as NSUInteger,
                                n,
                            );
                        }
                    }
                }
            }
        }
        unsafe {
            let _: () = msg0(encoder, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    /// Size the shadow arrays for this frame's casters, with their sampler
    /// and pipelines: a layer per directional and spot light, six per point
    /// light, at the size the lights of each kind ask for
    /// (`ShadowSettings::map_size`, the largest), shrunk when several share
    /// a kind (`shadow_layer_size`). Reallocated only to grow or resize.
    ///
    /// `map_size` used to be a field the wgpu path read and discarded; the
    /// failure mode there is worse than not offering it, since the caller sets
    /// 4096, sees no error and gets whatever was allocated. It is read here.
    fn ensure_shadow_arrays(&mut self, casters: &Casters) -> Result<(), MetalError> {
        use crate::lights::shadow::shadow_layer_size;
        let count = |want: fn(&Caster) -> bool| casters.iter().filter(|c| want(c)).map(Caster::layers).sum::<usize>();
        let nd = count(|c| matches!(c, Caster::Dir { .. }));
        let ns = count(|c| matches!(c, Caster::Spot { .. }));
        let np = count(|c| matches!(c, Caster::Point { .. })) / 6;
        let max_size = |want: fn(&Caster) -> bool| {
            casters.iter().filter(|c| want(c)).map(|c| c.settings().map_size).max().unwrap_or(0)
        };
        if nd + ns + np == 0 {
            return Ok(());
        }
        let array = |device: &MetalDevice, size: u32, layers: usize| {
            crate::metal::resources::new_texture_array_levels(
                device,
                size,
                size,
                pixel_format::DEPTH32_FLOAT,
                texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
                layers as u32,
                1,
            )
        };
        if nd > 0 {
            let size = shadow_layer_size(max_size(|c| matches!(c, Caster::Dir { .. })), nd);
            if self.shadow_map.is_none() || self.shadow_size != size || self.shadow_layers[0] < nd {
                self.shadow_map = Some(array(&self.device, size, nd)?);
                self.shadow_size = size;
                self.shadow_layers[0] = nd;
            }
        }
        if ns > 0 {
            let size = shadow_layer_size(max_size(|c| matches!(c, Caster::Spot { .. })), ns);
            if self.spot_shadow_map.is_none() || self.spot_shadow_size != size || self.shadow_layers[1] < ns {
                self.spot_shadow_map = Some(array(&self.device, size, ns)?);
                self.spot_shadow_size = size;
                self.shadow_layers[1] = ns;
            }
        }
        if np > 0 {
            // 512 a face, matching the wgpu path. Six faces make a cube
            // expensive to grow — 1024 is four times the memory and four times
            // the fill — and a point light's shadow is usually small in frame.
            if self.point_shadow_map.is_none() || self.shadow_layers[2] < 6 * np {
                self.point_shadow_map = Some(array(&self.device, POINT_SHADOW_SIZE, 6 * np)?);
                self.shadow_layers[2] = 6 * np;
            }
            if self.point_shadow_pipeline.is_none() {
                self.point_shadow_pipeline =
                    Some(crate::metal::resources::point_shadow_pipeline(&self.device)?);
            }
        }
        self.ensure_shadow_support()
    }

    /// The pipeline and comparison sampler every shadow pass shares.
    fn ensure_shadow_support(&mut self) -> Result<(), MetalError> {
        if self.shadow_sampler.is_none() {
            self.shadow_sampler = Some(crate::metal::resources::shadow_sampler(&self.device)?);
        }
        if self.shadow_pipeline.is_none() {
            self.shadow_pipeline = Some(crate::metal::resources::shadow_pipeline(&self.device)?);
        }
        if self.cut_shadow_pipeline.is_none() {
            self.cut_shadow_pipeline =
                Some(crate::metal::resources::cut_shadow_pipeline(&self.device)?);
        }
        Ok(())
    }

    /// Upload `scene.environment` if it is not the one already held.
    ///
    /// `scene.environment_cube_rt` — a live cube render target, fed by a
    /// `CubeCamera` — is not handled: it lives in the wgpu renderer's own
    /// resource table and this backend has no view of it. A scene using one
    /// gets no environment here rather than a wrong one.
    fn ensure_environment(&mut self, scene: &Scene) {
        // Prefiltered, as three.js's `WebGLCubeUVMaps` does for every
        // Standard/Physical environment.
        let cube = scene.environment.as_ref().map(|c| self.prefiltered(c));
        // three.js `scene.environmentIntensity` and `environmentRotation`.
        self.env_light = (
            scene.environment_intensity,
            crate::materials::environment_lookup_rows(
                scene.environment_rotation,
                cube.as_ref().is_some_and(|c| c.cube_uv_atlas.is_none()),
            ),
        );
        let Some(cube) = cube.as_ref() else {
            self.env_cube = None;
            return;
        };
        let key = crate::renderer::gpu_texture::cube_cache_key(cube);
        if matches!(&self.env_cube, Some((k, _)) if *k == key) {
            return;
        }
        let sk = SamplerKey::environment();
        if !self.samplers.contains_key(&sk) {
            if let Ok(smp) = new_sampler(&self.device, sk) {
                self.samplers.insert(sk, smp);
            }
        }
        let sk = SamplerKey::default_linear();
        if !self.samplers.contains_key(&sk) {
            if let Ok(smp) = new_sampler(&self.device, sk) {
                self.samplers.insert(sk, smp);
            }
        }
        match EnvGpu::upload(&self.device, cube) {
            Ok(env) => self.env_cube = Some((key, env)),
            Err(e) => {
                // A scene that cannot have its environment uploaded is better
                // rendered without one than not at all, but silently is how
                // this backend earned a reputation for being broken.
                eprintln!("threers: metal environment upload failed: {e:?}");
                self.env_cube = None;
            }
        }
    }

    /// three.js `WebGLCubeUVMaps`: a cube texture lighting a Standard or
    /// Physical material is read through its PMREM — prefiltered once and
    /// kept — unless it already carries one.
    fn prefiltered(&mut self, cube: &Arc<crate::textures::CubeTexture>) -> Arc<crate::textures::CubeTexture> {
        if cube.cube_uv_atlas.is_some() {
            return cube.clone();
        }
        self.pmrem_cache
            .entry(crate::renderer::gpu_texture::cube_cache_key(cube))
            .or_insert_with(|| crate::PmremGenerator::from_cube(cube.clone()))
            .clone()
    }

    /// [`prefiltered`](Self::prefiltered) once it has run for `cube`.
    fn prefiltered_cached(&self, cube: &Arc<crate::textures::CubeTexture>) -> Arc<crate::textures::CubeTexture> {
        self.pmrem_cache
            .get(&crate::renderer::gpu_texture::cube_cache_key(cube))
            .cloned()
            .unwrap_or_else(|| cube.clone())
    }

    /// The extra material maps, in the order the shader binds them:
    /// emissive, normal, metallic-roughness, ambient occlusion.
    ///
    /// glTF packs roughness and metalness into one image — roughness in green,
    /// metalness in blue — and `threers` keeps them as separate slots because
    /// three.js does. When both point at the same texture, which is what a glTF
    /// load produces, one binding carries both. When they differ, roughness
    /// wins the slot and metalness falls back to its scalar, because there is
    /// no third channel to put it in and the alternative is a sixth binding for
    /// a case no real asset produces.
    ///
    /// Each with its own sampler, from the texture's own filter and wrap.
    fn material_maps(&self, material: &Material) -> ([Option<(Id, Id)>; 4], [u32; 4]) {
        let slots = material.texture_slots();
        let mr = slots.roughness_map.as_ref().or(slots.metalness_map.as_ref());
        let ids = [
            self.slot_texture(slots.emissive_map.as_ref(), false),
            self.slot_texture(slots.normal_map.as_ref(), true),
            self.slot_texture(mr, true),
            self.slot_texture(slots.ao_map.as_ref(), true),
        ];
        let present = std::array::from_fn(|i| ids[i].is_some() as u32);
        (ids, present)
    }

    /// Build the `MTLRenderPassDescriptor` for one view. Autoreleased.
    ///
    /// `clear` is false for a second view sharing a target with the first —
    /// side-by-side eyes in one texture, where clearing again would erase the
    /// eye already drawn.
    unsafe fn render_pass_descriptor(
        &self,
        pass: &PassAttachments,
        scene: &Scene,
        view: &RenderView,
        clear: bool,
        keep_multisampled: bool,
        array_length: u32,
    ) -> Id {
        let rpd: Id = msg0(
            class!("MTLRenderPassDescriptor"),
            sel!("renderPassDescriptor"),
        );
        // A layered pass renders into every slice at once, so it names how many
        // there are and starts at slice 0 — the shader's
        // `render_target_array_index` picks between them. Without this the
        // attachment is one slice deep and both eyes land on top of each other.
        if array_length > 1 {
            let _: () = msg1(
                rpd,
                sel!("setRenderTargetArrayLength:"),
                array_length as NSUInteger,
            );
        }
        let slice = if array_length > 1 { 0 } else { view.slice };
        let color_attachments: Id = msg0(rpd, sel!("colorAttachments"));
        let color: Id = msg1(color_attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let color_texture = if view.color.is_null() {
            pass.color
        } else {
            view.color
        };
        let _: () = msg1(color, sel!("setTexture:"), color_texture);
        let _: () = msg1(color, sel!("setSlice:"), slice as NSUInteger);
        let _: () = msg1(
            color,
            sel!("setLoadAction:"),
            if clear {
                load_action::CLEAR
            } else {
                load_action::LOAD
            },
        );
        let _: () = msg1(
            color,
            sel!("setClearColor:"),
            clear_color(scene.background, scene.background_alpha),
        );
        if pass.resolve.is_null() {
            let _: () = msg1(color, sel!("setStoreAction:"), store_action::STORE);
        } else {
            let _: () = msg1(color, sel!("setResolveTexture:"), pass.resolve);
            let _: () = msg1(color, sel!("setResolveSlice:"), slice as NSUInteger);
            let _: () = msg1(
                color,
                sel!("setStoreAction:"),
                if keep_multisampled {
                    store_action::STORE_AND_MULTISAMPLE_RESOLVE
                } else {
                    store_action::MULTISAMPLE_RESOLVE
                },
            );
        }

        let depth_texture = if view.depth.is_null() {
            pass.depth
        } else {
            view.depth
        };
        if !depth_texture.is_null() {
            let depth: Id = msg0(rpd, sel!("depthAttachment"));
            let _: () = msg1(depth, sel!("setTexture:"), depth_texture);
            let _: () = msg1(depth, sel!("setSlice:"), slice as NSUInteger);
            // Always cleared, even for a view that loads its colour: depth is
            // usually not stored (below), so loading it reads undefined values
            // across the second view's whole viewport. Views sharing a target
            // do not overlap — or, if they do, the later one is an overlay —
            // so nothing needs the earlier view's depth.
            let _: () = msg1(depth, sel!("setLoadAction:"), load_action::CLEAR);
            // Normally discarded: depth is scratch for the pass that draws it,
            // and storing it costs bandwidth nothing reads. SSAO does read it
            // afterwards, and a discarded depth texture samples as the clear
            // value — which sends every pixel down the far-plane early-out and
            // produces occlusion that is uniformly, silently absent.
            // Under MSAA, stored depth is also resolved where the target
            // asks (`PassAttachments::with_depth_resolve`) — the multisampled
            // one kept as well while the OIT pass still has to test against
            // it.
            let resolve_depth =
                self.store_depth && view.depth.is_null() && !pass.depth_resolve.is_null() && pass.sample_count > 1;
            if resolve_depth {
                let _: () = msg1(depth, sel!("setResolveTexture:"), pass.depth_resolve);
                let _: () = msg1(depth, sel!("setResolveSlice:"), slice as NSUInteger);
            }
            let _: () = msg1(
                depth,
                sel!("setStoreAction:"),
                match (self.store_depth, resolve_depth, self.oit_active) {
                    (false, ..) => store_action::DONT_CARE,
                    (true, false, _) => store_action::STORE,
                    (true, true, false) => store_action::MULTISAMPLE_RESOLVE,
                    (true, true, true) => store_action::STORE_AND_MULTISAMPLE_RESOLVE,
                },
            );
            // Reverse-Z clears to the far plane, which is 0.
            let _: () = msg1(
                depth,
                sel!("setClearDepth:"),
                if pass.reverse_z { 0.0f64 } else { 1.0f64 },
            );
        }
        rpd
    }
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
unsafe fn set_bytes<T>(encoder: Id, vertex_stage: bool, value: &T, index: NSUInteger) {
    debug_assert!(std::mem::size_of::<T>() <= 4096);
    let sel = if vertex_stage {
        sel!("setVertexBytes:length:atIndex:")
    } else {
        sel!("setFragmentBytes:length:atIndex:")
    };
    let _: () = msg3(
        encoder,
        sel,
        value as *const T as *const c_void,
        std::mem::size_of::<T>() as NSUInteger,
        index,
    );
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
            for tri in idx.chunks_exact(3) {
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
            // and two `map_xform` rows a map — the members `struct Draw`
            // declares in `shaders.metal`.
            64 + 64 + 9 * 16 + 6 * 16 + 6 * 16 + 7 * 16 + 16 + 2 * MAP_COUNT * 16
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
