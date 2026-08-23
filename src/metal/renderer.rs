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
    new_sampler, render_pipeline, upload_texture, white_texture, PipelineKey, SamplerKey,
};
use crate::cameras::Camera;
use crate::core::{BufferGeometry, Layers, ObjectKind};
use crate::lights::Light;
use crate::materials::{Material, MaterialKind};
use crate::math::{Matrix4, Vector3};
use crate::scene::Scene;
use crate::textures::Texture;

/// Views (eyes) one frame can carry, matching `MAX_VIEWS` in `shaders.metal`.
/// Two: a plain camera uses one, a headset both eyes.
pub const MAX_VIEWS: usize = 2;

/// Light counts, matching the array sizes in `shaders.metal`.
const MAX_DIR_LIGHTS: usize = 4;
const MAX_POINT_LIGHTS: usize = 8;
const MAX_SPOT_LIGHTS: usize = 4;
const MAX_HEMI_LIGHTS: usize = 2;

/// Buffer indices, matching the `[[buffer(n)]]` attributes in `shaders.metal`.
const VB_VERTICES: NSUInteger = 0;
const VB_FRAME: NSUInteger = 1;
const VB_DRAW: NSUInteger = 2;
const VB_INSTANCES: NSUInteger = 3;
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
    pub _pad: f32,
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
}

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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct DrawUniforms {
    model: [f32; 16],
    normal_mat: [f32; 16],
    base_color: [f32; 4],
    emissive: [f32; 4],
    specular: [f32; 4],
    pbr: [f32; 4],
    uv_transform: [f32; 4],
    misc: [f32; 4],
    flags: [u32; 4],
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
            uv_transform: [0.0, 0.0, 1.0, 1.0],
            misc: [0.0, 1.0, 0.1, 1000.0],
            flags: [0; 4],
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
            drawable,
            width,
            height,
            color_format,
            depth_format,
            sample_count,
            reverse_z: false,
        }
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
    /// Geometries uploaded this frame (a cache miss or a version bump).
    pub geometry_uploads: u32,
    /// Textures uploaded this frame.
    pub texture_uploads: u32,
}

// -------------------------------------------------------------------- caches

struct CachedGeometry {
    /// Held to pin the allocation: the cache key is the `Arc`'s address, and a
    /// dropped geometry's address can otherwise be handed to a new one.
    _geometry: Arc<BufferGeometry>,
    version: u32,
    vertices: Owned,
    indices: Option<Owned>,
    index_count: usize,
    vertex_count: usize,
    last_used: u64,
}

struct CachedTexture {
    _texture: Arc<Texture>,
    texture: Owned,
    sampler_key: SamplerKey,
    last_used: u64,
}

/// One item collected from the scene graph, ready to encode.
struct DrawItem {
    geometry: Arc<BufferGeometry>,
    material: Arc<Material>,
    world: Matrix4,
    primitive: NSUInteger,
    /// Per-instance matrices, until they are uploaded — then the index of the
    /// buffer holding them, so a skipped item cannot shift another item's
    /// buffer out from under it.
    instances: Option<Vec<f32>>,
    instance_slot: Option<usize>,
    instance_count: usize,
    view_depth: f32,
    render_order: i32,
    transparent: bool,
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
/// Not here, and handled by [`crate::renderer::Renderer`] instead: shadow maps,
/// post-processing, environment/IBL, skinning, morph targets, and the normal /
/// roughness / metalness / AO / emissive map slots. Instanced meshes with
/// non-uniform per-instance scale shade with skewed normals.
pub struct MetalRenderer {
    device: MetalDevice,
    pipelines: HashMap<PipelineKey, Owned>,
    samplers: HashMap<SamplerKey, Owned>,
    geometries: HashMap<usize, CachedGeometry>,
    textures: HashMap<usize, CachedTexture>,
    /// `[forward-write, forward-read, reverse-write, reverse-read]`.
    depth_states: [Owned; 4],
    white: Owned,
    identity_instance: Owned,
    transients: Vec<Vec<Owned>>,
    frame: u64,
    cache_retention: u64,
}

impl MetalRenderer {
    /// Acquire the default device and build the renderer.
    pub fn new() -> Result<Self, MetalError> {
        Self::with_device(MetalDevice::new()?)
    }

    /// Build on an existing device — use this to share one device (and one
    /// compiled shader library) between a window and an offscreen target.
    pub fn with_device(device: MetalDevice) -> Result<Self, MetalError> {
        let depth_states = [
            depth_stencil_state(&device, compare::LESS, true)?,
            depth_stencil_state(&device, compare::LESS, false)?,
            depth_stencil_state(&device, compare::GREATER, true)?,
            depth_stencil_state(&device, compare::GREATER, false)?,
        ];
        let white = white_texture(&device)?;
        let identity_instance =
            device.new_buffer(bytemuck::cast_slice(&[Matrix4::identity().elements]))?;
        Ok(Self {
            device,
            pipelines: HashMap::new(),
            samplers: HashMap::new(),
            geometries: HashMap::new(),
            textures: HashMap::new(),
            depth_states,
            white,
            identity_instance,
            transients: (0..FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect(),
            frame: 0,
            cache_retention: DEFAULT_CACHE_RETENTION,
        })
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
        self.frame = self.frame.wrapping_add(1);
        let slot = (self.frame % FRAMES_IN_FLIGHT as u64) as usize;
        self.transients[slot].clear();

        scene.update_world();
        let (frame_u, mut items, mut stats) = self.collect(scene, views, pass);

        // Opaque front-to-back (early-z rejects the overdraw), transparent
        // back-to-front (blending is order-dependent). `render_order` wins over
        // both, as in three.js.
        items.sort_by(|a, b| {
            a.transparent
                .cmp(&b.transparent)
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
            if self.ensure_geometry(&item.geometry)? {
                stats.geometry_uploads += 1;
            }
            let slots = item.material.texture_slots();
            for map in [slots.map, slots.matcap_map].into_iter().flatten() {
                if self.ensure_texture(&map)? {
                    stats.texture_uploads += 1;
                }
            }
            if let Some(data) = item.instances.take() {
                let buf = self.device.new_buffer(bytemuck::cast_slice(&data))?;
                item.instance_count = data.len() / 16;
                item.instance_slot = Some(self.transients[slot].len());
                self.transients[slot].push(buf);
            }
        }

        unsafe { self.encode(pass, views, &frame_u, &items, scene, slot, &mut stats) }?;
        self.evict_unused();
        Ok(stats)
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
    ) -> (FrameUniforms, Vec<DrawItem>, MetalRenderStats) {
        // View 0 is the one the traversal uses: its layer mask decides what is
        // drawn and its position orders the transparent items. Two eyes a few
        // centimetres apart do not disagree about either.
        let primary = &views[0];
        let eye = primary.position;
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
            view_info: [views.len() as u32, 0, 0, 0],
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

        let mut items = Vec::new();
        let mut stats = MetalRenderStats::default();
        let (mut n_dir, mut n_point, mut n_spot, mut n_hemi) = (0usize, 0, 0, 0);
        let cam_layers = primary.layers;

        scene.arena.traverse_visible(scene.root, &mut |_id, obj| {
            if !cam_layers.test(&obj.layers) {
                return;
            }
            let world = obj.matrix_world;
            let mut push = |geometry: &Arc<BufferGeometry>,
                            material: &Arc<Material>,
                            primitive: NSUInteger,
                            instances: Option<Vec<f32>>| {
                if !geometry.attributes.contains_key("position") {
                    stats.skipped += 1;
                    return;
                }
                let center =
                    Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
                items.push(DrawItem {
                    geometry: geometry.clone(),
                    material: material.clone(),
                    world,
                    primitive,
                    instance_count: instances.as_ref().map(|v| v.len() / 16).unwrap_or(1),
                    instances,
                    instance_slot: None,
                    view_depth: center.distance_to(eye),
                    render_order: obj.render_order,
                    transparent: is_transparent(material),
                });
            };

            match &obj.kind {
                ObjectKind::Mesh(m) => push(&m.geometry, &m.material, primitive::TRIANGLE, None),
                ObjectKind::SkinnedMesh(sm) => {
                    // Drawn in its bind pose: skinning is not in this backend.
                    push(&sm.geometry, &sm.material, primitive::TRIANGLE, None)
                }
                ObjectKind::LineSegments(ls) => {
                    push(&ls.geometry, &ls.material, primitive::LINE, None)
                }
                ObjectKind::Points(p) => push(&p.geometry, &p.material, primitive::POINT, None),
                ObjectKind::Sprite(s) => {
                    if let Some(g) = sprite_geometry(s) {
                        push(&g, &s.material, primitive::TRIANGLE, None);
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
                    push(&im.geometry, &im.material, primitive::TRIANGLE, Some(data));
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
                        n_point += 1;
                    }
                    Light::Spot(l) => {
                        if n_spot >= MAX_SPOT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let dir = transform_direction(&world, l.direction).normalize();
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
                    // RectAreaLight has no analytic form in this backend.
                    Light::RectArea(_) => {}
                },
                ObjectKind::Group => {}
            }
        });

        frame_u.counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        (frame_u, items, stats)
    }

    // --------------------------------------------------------------- uploads

    /// Ensure `geometry` has current GPU buffers. `true` if it uploaded.
    fn ensure_geometry(&mut self, geometry: &Arc<BufferGeometry>) -> Result<bool, MetalError> {
        let key = Arc::as_ptr(geometry) as usize;
        let frame = self.frame;
        if let Some(cached) = self.geometries.get_mut(&key) {
            if cached.version == geometry.geometry_version {
                cached.last_used = frame;
                return Ok(false);
            }
        }
        let vertices = build_vertices(geometry);
        if vertices.is_empty() {
            return Ok(false);
        }
        let vbuf = self.device.new_buffer(bytemuck::cast_slice(&vertices))?;
        let (ibuf, index_count) = match &geometry.index {
            Some(idx) if !idx.is_empty() => (
                Some(self.device.new_buffer(bytemuck::cast_slice(idx))?),
                idx.len(),
            ),
            _ => (None, 0),
        };
        self.geometries.insert(
            key,
            CachedGeometry {
                _geometry: geometry.clone(),
                version: geometry.geometry_version,
                vertices: vbuf,
                indices: ibuf,
                index_count,
                vertex_count: vertices.len(),
                last_used: frame,
            },
        );
        Ok(true)
    }

    /// Ensure `texture` is on the GPU with a matching sampler. `true` if it uploaded.
    fn ensure_texture(&mut self, texture: &Arc<Texture>) -> Result<bool, MetalError> {
        let key = Arc::as_ptr(texture) as usize;
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
        let Ok(gpu) = upload_texture(&self.device, texture) else {
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
    ) -> Result<(), MetalError> {
        let cmd = self.device.command_buffer();
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }

        if layered(views, pass) {
            let descriptor =
                self.render_pass_descriptor(pass, scene, &views[0], true, views.len() as u32);
            self.encode_pass(
                cmd,
                descriptor,
                pass,
                frame_u,
                views[0].viewport,
                items,
                slot,
                stats,
                views.len() as u32,
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
                let clear = !cleared.contains(&target);
                cleared.push(target);

                // The mono shaders read view 0, so the view being drawn moves
                // into that slot for this pass.
                let mut per_view = *frame_u;
                per_view.promote_view(index);
                let descriptor = self.render_pass_descriptor(pass, scene, view, clear, 1);
                self.encode_pass(
                    cmd,
                    descriptor,
                    pass,
                    &per_view,
                    view.viewport,
                    items,
                    slot,
                    stats,
                    1,
                )?;
            }
        }

        if !pass.drawable.is_null() {
            let _: () = msg1(cmd, sel!("presentDrawable:"), pass.drawable);
        }
        let _: () = msg0(cmd, sel!("commit"));
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
        items: &[DrawItem],
        slot: usize,
        stats: &mut MetalRenderStats,
        views_per_draw: u32,
    ) -> Result<(), MetalError> {
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
        if let Some(viewport) = viewport {
            let _: () = msg1(encoder, sel!("setViewport:"), viewport);
        }
        set_bytes(encoder, true, frame_u, VB_FRAME);
        set_bytes(encoder, false, frame_u, FB_FRAME);

        let mut last_pipeline = NIL;
        let mut last_cull = usize::MAX;
        let mut last_fill = usize::MAX;
        let mut last_depth = NIL;
        // A failure inside the loop still has to close the encoder: Metal
        // aborts the process if one is released without `endEncoding`.
        let mut failure = None;

        for item in items {
            let key = Arc::as_ptr(&item.geometry) as usize;
            // Nothing uploaded for this geometry — an attribute set the vertex
            // builder could make nothing of. Counted, not drawn, not fatal.
            let Some(geom) = self.geometries.get(&key) else {
                stats.skipped += 1;
                continue;
            };
            let (vertices, indices, index_count, vertex_count) = (
                geom.vertices.id(),
                geom.indices.as_ref().map(|b| b.id()),
                geom.index_count,
                geom.vertex_count,
            );

            let material = &item.material;
            let point_sprites = item.primitive == primitive::POINT;
            let layered_draw = views_per_draw > 1;
            let pipeline = match self.pipeline(PipelineKey {
                point_sprites,
                blend: item.transparent,
                layered: layered_draw,
                // Only the layered path needs this, and naming it there costs a
                // pipeline per topology — so leave it unspecified elsewhere.
                topology: if layered_draw {
                    topology_class::of(item.primitive)
                } else {
                    topology_class::UNSPECIFIED
                },
                color_format: pass.color_format,
                depth_format: pass.depth_format,
                sample_count: pass.sample_count.max(1),
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

            // Transparent surfaces test depth but do not write it, so two
            // blended layers both survive rather than the nearer one erasing
            // the further one's contribution. A pass with no depth attachment
            // gets no depth state at all: Metal rejects one that would write.
            if !pass.depth.is_null() {
                let depth_state =
                    self.depth_states[pass.reverse_z as usize * 2 + item.transparent as usize].id();
                if depth_state != last_depth {
                    let _: () = msg1(encoder, sel!("setDepthStencilState:"), depth_state);
                    last_depth = depth_state;
                }
            }

            let cull = match material.side() {
                1 => cull::FRONT,
                2 => cull::NONE,
                _ => cull::BACK,
            };
            if cull != last_cull {
                let _: () = msg1(encoder, sel!("setCullMode:"), cull);
                last_cull = cull;
            }
            let fill = if material.wireframe() {
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

            let (texture, sampler) = self.material_texture(material);
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                texture,
                0usize,
            );
            let _: () = msg2(
                encoder,
                sel!("setFragmentSamplerState:atIndex:"),
                sampler,
                0usize,
            );

            let draw_u = draw_uniforms(
                item,
                texture != self.white.id(),
                instanced.is_some(),
                &item.geometry,
            );
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
            match indices {
                Some(ibuf) if index_count > 0 => {
                    let _: () = msg6(
                        encoder,
                        sel!(
                            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                        ),
                        item.primitive,
                        index_count as NSUInteger,
                        index_type::UINT32,
                        ibuf,
                        0usize,
                        instances as NSUInteger,
                    );
                }
                _ => {
                    let _: () = msg4(
                        encoder,
                        sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                        item.primitive,
                        0usize,
                        vertex_count as NSUInteger,
                        instances as NSUInteger,
                    );
                }
            }

            stats.draw_calls += 1;
            if item.primitive == primitive::TRIANGLE {
                let verts = if index_count > 0 {
                    index_count
                } else {
                    vertex_count
                };
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
        let map = match material.kind() {
            MaterialKind::Matcap => slots.matcap_map.as_ref().or(slots.map.as_ref()),
            _ => slots.map.as_ref(),
        };
        if let Some(map) = map {
            if let Some(cached) = self.textures.get(&(Arc::as_ptr(map) as usize)) {
                if let Some(sampler) = self.samplers.get(&cached.sampler_key) {
                    return (cached.texture.id(), sampler.id());
                }
            }
        }
        let sampler = self
            .samplers
            .get(&SamplerKey::default_linear())
            .map(|s| s.id())
            .unwrap_or(NIL);
        (self.white.id(), sampler)
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
            MTLClearColor {
                red: scene.background.r as f64,
                green: scene.background.g as f64,
                blue: scene.background.b as f64,
                alpha: scene.background_alpha as f64,
            },
        );
        if pass.resolve.is_null() {
            let _: () = msg1(color, sel!("setStoreAction:"), store_action::STORE);
        } else {
            let _: () = msg1(color, sel!("setResolveTexture:"), pass.resolve);
            let _: () = msg1(color, sel!("setResolveSlice:"), slice as NSUInteger);
            let _: () = msg1(
                color,
                sel!("setStoreAction:"),
                store_action::MULTISAMPLE_RESOLVE,
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
            let _: () = msg1(
                depth,
                sel!("setLoadAction:"),
                if clear {
                    load_action::CLEAR
                } else {
                    load_action::LOAD
                },
            );
            let _: () = msg1(depth, sel!("setStoreAction:"), store_action::DONT_CARE);
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

/// A material draws blended when it says so, or when it is not fully opaque.
fn is_transparent(material: &Material) -> bool {
    material.transparent() || material.opacity() < 1.0
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

/// Per-draw uniforms for one item.
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
    let (offset, repeat, rotation) = match slots.map.as_ref() {
        Some(map) => (map.offset, map.repeat, map.rotation),
        None => (
            crate::math::Vector2::ZERO,
            crate::math::Vector2::ONE,
            0.0f32,
        ),
    };
    DrawUniforms {
        model: item.world.elements,
        normal_mat: normal_matrix(&item.world),
        base_color: [color.r, color.g, color.b, material.opacity()],
        emissive: [emissive.r, emissive.g, emissive.b, 0.0],
        specular: [specular.r, specular.g, specular.b, material.shininess()],
        pbr: [
            material.roughness(),
            material.metalness(),
            material.alpha_test(),
            material.toon_steps() as f32,
        ],
        uv_transform: [offset.x, offset.y, repeat.x, repeat.y],
        misc: [
            rotation,
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
            instanced as u32,
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
        out.push(Vertex {
            position: [
                positions.array[p],
                positions.array[p + 1],
                positions.array[p + 2],
            ],
            normal,
            uv,
            color,
            _pad: 0.0,
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
/// a shared unit quad on the XY plane. It is not billboarded — the scene
/// graph's own transform orients it — which is the honest limit of drawing
/// sprites without a dedicated pipeline.
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
        assert_eq!(std::mem::size_of::<DrawUniforms>(), 240);
        assert_eq!(
            std::mem::size_of::<FrameUniforms>(),
            // Per view: view_proj, view, camera_pos. Then 6 float4s of frame
            // state, then the light arrays.
            MAX_VIEWS * (64 + 64 + 16) + 16 * 6 + 4 * 32 + 8 * 48 + 4 * 64 + 2 * 48
        );
        // Both fit the 4 KB `setVertexBytes` limit.
        assert!(std::mem::size_of::<FrameUniforms>() <= 4096);
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
    fn transparency_follows_flag_or_opacity() {
        use crate::materials::BasicMaterial;
        let opaque = Material::Basic(BasicMaterial::default());
        assert!(!is_transparent(&opaque));
        let faded = Material::Basic(BasicMaterial::default().with_opacity(0.5));
        assert!(is_transparent(&faded));
        let flagged = Material::Basic(BasicMaterial::default().with_transparent(true));
        assert!(is_transparent(&flagged));
    }
}
