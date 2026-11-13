//! The visionOS frame loop: CompositorServices and ARKit, declared and driven.
//!
//! Compiled only for `target_os = "visionos"`. Every signature here is
//! transcribed from the XROS SDK headers — `CompositorServices/drawable.h`,
//! `view.h`, `frame.h`, `layer_renderer.h`, and `ARKit/world_tracking.h`.

use std::ffi::c_void;

use super::super::device::{MetalDevice, MetalError};
use super::super::enums::{MTLViewport, NSUInteger};
use super::super::objc::{msg0, sel, AutoreleasePool, Id, NIL};
use super::super::renderer::{MetalRenderStats, MetalRenderer, PassAttachments, RenderView};
use super::{
    ArDataProviders, ArDeviceAnchor, ArSession, ArWorldTrackingConfiguration,
    ArWorldTrackingProvider, AxisConvention, CpDrawable, CpFrame, CpFrameTiming, CpLayerRenderer,
    CpLayerRendererProperties, CpTime, CpView, CpViewTextureMap, LayerRendererState, SimdFloat2,
    SimdFloat4x4,
};
use crate::math::Matrix4;
use crate::scene::Scene;

// Apple's ARM64 convention returns a `simd_float4x4` (four vector columns)
// in `q0`–`q3`, and takes a `simd_float2` in one vector register. Stable
// Rust has no type that FFI passes that way: a `#[repr(C)]` struct of
// sixteen floats, or of four NEON vectors, comes back through memory
// (`x8`), and a pair of floats goes in two registers. Read that way, every
// pose and projection was whatever the registers held, and
// `cp_drawable_set_depth_range` saw a near plane of 0. So these few calls
// are made as C makes them: the matrix ones by a branch in assembly that
// reads `q0`–`q3` back, the pair as an `f64` with both floats in its bits
// (it lands in `d0`, the low half of the register the callee reads).

/// Calls `f(x0, x1, x2)`, a C function returning a `simd_float4x4`.
pub(crate) unsafe fn call_returning_float4x4(f: unsafe extern "C" fn(), x0: usize, x1: usize, x2: usize) -> SimdFloat4x4 {
    use std::arch::aarch64::float32x4_t;
    let (c0, c1, c2, c3): (float32x4_t, float32x4_t, float32x4_t, float32x4_t);
    // SAFETY: `f` takes up to three integer arguments and returns the four
    // columns in v0–v3, per the platform's convention; everything else the C
    // convention lets it clobber is declared clobbered.
    unsafe {
        std::arch::asm!(
            "blr {f}",
            f = in(reg) f,
            inout("x0") x0 => _,
            inout("x1") x1 => _,
            inout("x2") x2 => _,
            out("v0") c0,
            out("v1") c1,
            out("v2") c2,
            out("v3") c3,
            clobber_abi("C"),
        );
        SimdFloat4x4 { columns: std::mem::transmute::<[float32x4_t; 4], [[f32; 4]; 4]>([c0, c1, c2, c3]) }
    }
}

/// `simd_float2` in an `f64`'s bits: `x` in the low lane, `y` in the high.
fn float2_bits(v: SimdFloat2) -> f64 {
    f64::from_bits(((v.y.to_bits() as u64) << 32) | v.x.to_bits() as u64)
}

pub(crate) unsafe fn cp_drawable_set_depth_range(drawable: CpDrawable, depth_range: SimdFloat2) {
    unsafe { cp_drawable_set_depth_range_raw(drawable, float2_bits(depth_range)) }
}

pub(crate) unsafe fn cp_drawable_compute_projection(drawable: CpDrawable, convention: u8, view_index: NSUInteger) -> SimdFloat4x4 {
    unsafe { call_returning_float4x4(cp_drawable_compute_projection_sym, drawable as usize, convention as usize, view_index) }
}

pub(crate) unsafe fn cp_view_get_transform(view: CpView) -> SimdFloat4x4 {
    unsafe { call_returning_float4x4(cp_view_get_transform_sym, view as usize, 0, 0) }
}

pub(crate) unsafe fn ar_device_anchor_get_origin_from_anchor_transform(anchor: ArDeviceAnchor) -> SimdFloat4x4 {
    unsafe { call_returning_float4x4(ar_device_anchor_get_origin_from_anchor_transform_sym, anchor as usize, 0, 0) }
}

#[link(name = "CompositorServices", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn cp_layer_renderer_get_state(layer_renderer: CpLayerRenderer) -> u32;
    pub(crate) fn cp_layer_renderer_wait_until_running(layer_renderer: CpLayerRenderer);
    pub(crate) fn cp_layer_renderer_query_next_frame(layer_renderer: CpLayerRenderer) -> CpFrame;
    pub(crate) fn cp_layer_renderer_get_device(layer_renderer: CpLayerRenderer) -> Id;
    pub(crate) fn cp_layer_renderer_get_properties(
        layer_renderer: CpLayerRenderer,
    ) -> CpLayerRendererProperties;
    pub(crate) fn cp_layer_renderer_properties_get_view_count(
        properties: CpLayerRendererProperties,
    ) -> NSUInteger;

    pub(crate) fn cp_frame_predict_timing(frame: CpFrame) -> CpFrameTiming;
    pub(crate) fn cp_frame_start_update(frame: CpFrame);
    pub(crate) fn cp_frame_end_update(frame: CpFrame);
    pub(crate) fn cp_frame_start_submission(frame: CpFrame);
    pub(crate) fn cp_frame_end_submission(frame: CpFrame);
    pub(crate) fn cp_frame_query_drawable(frame: CpFrame) -> CpDrawable;

    pub(crate) fn cp_frame_timing_get_optimal_input_time(timing: CpFrameTiming) -> CpTime;
    pub(crate) fn cp_frame_timing_get_presentation_time(timing: CpFrameTiming) -> CpTime;
    pub(crate) fn cp_time_wait_until(time: CpTime);
    pub(crate) fn cp_time_to_cf_time_interval(time: CpTime) -> f64;

    pub(crate) fn cp_drawable_get_view_count(drawable: CpDrawable) -> NSUInteger;
    pub(crate) fn cp_drawable_get_view(drawable: CpDrawable, index: NSUInteger) -> CpView;
    pub(crate) fn cp_drawable_get_color_texture(drawable: CpDrawable, index: NSUInteger) -> Id;
    pub(crate) fn cp_drawable_get_depth_texture(drawable: CpDrawable, index: NSUInteger) -> Id;
    pub(crate) fn cp_drawable_get_frame_timing(drawable: CpDrawable) -> CpFrameTiming;
    pub(crate) fn cp_drawable_set_device_anchor(drawable: CpDrawable, device_anchor: ArDeviceAnchor);
    #[link_name = "cp_drawable_set_depth_range"]
    fn cp_drawable_set_depth_range_raw(drawable: CpDrawable, depth_range: f64);
    // Called through `call_returning_float4x4`: (drawable, convention, view index).
    #[link_name = "cp_drawable_compute_projection"]
    fn cp_drawable_compute_projection_sym();
    pub(crate) fn cp_drawable_encode_present(drawable: CpDrawable, command_buffer: Id);

    #[link_name = "cp_view_get_transform"]
    fn cp_view_get_transform_sym();
    pub(crate) fn cp_view_get_view_texture_map(view: CpView) -> CpViewTextureMap;
    pub(crate) fn cp_view_texture_map_get_texture_index(map: CpViewTextureMap) -> NSUInteger;
    pub(crate) fn cp_view_texture_map_get_slice_index(map: CpViewTextureMap) -> NSUInteger;
    pub(crate) fn cp_view_texture_map_get_viewport(map: CpViewTextureMap) -> MTLViewport;
}

#[link(name = "ARKit", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn ar_session_create() -> ArSession;
    pub(crate) fn ar_session_run(session: ArSession, data_providers: ArDataProviders);
    pub(crate) fn ar_world_tracking_configuration_create() -> ArWorldTrackingConfiguration;
    pub(crate) fn ar_world_tracking_provider_create(
        configuration: ArWorldTrackingConfiguration,
    ) -> ArWorldTrackingProvider;
    pub(crate) fn ar_world_tracking_provider_query_device_anchor_at_timestamp(
        provider: ArWorldTrackingProvider,
        timestamp: f64,
        device_anchor: ArDeviceAnchor,
    ) -> isize;
    pub(crate) fn ar_data_providers_create() -> ArDataProviders;
    pub(crate) fn ar_data_providers_add_data_provider(providers: ArDataProviders, provider: *mut c_void);
    pub(crate) fn ar_device_anchor_create() -> ArDeviceAnchor;
    #[link_name = "ar_device_anchor_get_origin_from_anchor_transform"]
    fn ar_device_anchor_get_origin_from_anchor_transform_sym();
    pub(crate) fn ar_release(object: *mut c_void);
}

/// An ARKit object, released on drop. ARKit's C objects are `os_object`s, so
/// `ar_release` is the counterpart to every `*_create`.
struct ArObject(*mut c_void);

impl ArObject {
    fn new(ptr: *mut c_void) -> Option<Self> {
        (!ptr.is_null()).then_some(Self(ptr))
    }
    fn ptr(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for ArObject {
    fn drop(&mut self) {
        unsafe { ar_release(self.0) }
    }
}

// ------------------------------------------------------------ world tracking

/// ARKit world tracking: where the headset is, in the world.
///
/// Without it the eye transforms are relative to the device itself and the scene
/// is head-locked — it follows the viewer instead of staying put. This asks
/// ARKit for the device anchor at the frame's predicted presentation time, which
/// is what makes content hold still in the room.
pub struct WorldTracking {
    _session: ArObject,
    provider: ArObject,
    anchor: ArObject,
}

impl WorldTracking {
    /// Start a session with a world-tracking provider.
    ///
    /// The app must carry `NSWorldSensingUsageDescription` in its Info.plist;
    /// without it ARKit refuses the provider and every query fails, leaving the
    /// scene head-locked rather than crashing.
    pub fn new() -> Result<Self, MetalError> {
        Self::with_providers(&[])
    }

    /// [`new`](Self::new), with more data providers (hand tracking) run in
    /// the same session.
    pub(crate) fn with_providers(extra: &[*mut c_void]) -> Result<Self, MetalError> {
        unsafe {
            let configuration = ArObject::new(ar_world_tracking_configuration_create())
                .ok_or_else(|| MetalError::Unsupported("ar_world_tracking_configuration".into()))?;
            let provider = ArObject::new(ar_world_tracking_provider_create(configuration.ptr()))
                .ok_or_else(|| MetalError::Unsupported("ar_world_tracking_provider".into()))?;
            let providers = ArObject::new(ar_data_providers_create())
                .ok_or_else(|| MetalError::Unsupported("ar_data_providers".into()))?;
            ar_data_providers_add_data_provider(providers.ptr(), provider.ptr());
            for &p in extra {
                ar_data_providers_add_data_provider(providers.ptr(), p);
            }

            let session = ArObject::new(ar_session_create())
                .ok_or_else(|| MetalError::Unsupported("ar_session".into()))?;
            ar_session_run(session.ptr(), providers.ptr());

            let anchor = ArObject::new(ar_device_anchor_create())
                .ok_or_else(|| MetalError::Unsupported("ar_device_anchor".into()))?;
            Ok(Self {
                _session: session,
                provider,
                anchor,
            })
        }
    }

    /// The device pose at `timestamp` (seconds, `CFTimeInterval`), or `None`
    /// while tracking is still coming up.
    ///
    /// Returns `origin_from_device`: the transform from device space into world
    /// space.
    pub fn device_pose(&self, timestamp: f64) -> Option<Matrix4> {
        unsafe {
            let status = ar_world_tracking_provider_query_device_anchor_at_timestamp(
                self.provider.ptr(),
                timestamp,
                self.anchor.ptr(),
            );
            (status == 0).then(|| {
                ar_device_anchor_get_origin_from_anchor_transform(self.anchor.ptr()).into()
            })
        }
    }

    /// The `ar_device_anchor_t` this tracker fills in, to hand back to the
    /// compositor so it can reproject the frame.
    pub(crate) fn anchor_ptr(&self) -> ArDeviceAnchor {
        self.anchor.ptr()
    }
}

// ------------------------------------------------------------------ renderer

/// The nearest near plane the compositor accepts, metres.
const MIN_NEAR: f32 = 0.1;

/// A frame of an immersive space, as [`ImmersiveRenderer::render_frame_with`]
/// hands it to an update.
pub struct ImmersiveFrame<'a> {
    /// The head (the device) in the space, at the frame's predicted
    /// presentation time: `None` without world tracking, when the scene is
    /// head-locked.
    pub head: Option<Matrix4>,
    /// The eyes about to be drawn.
    pub views: &'a [RenderView],
    /// The drawable's size, in pixels.
    pub size: (u32, u32),
}

/// Draws a [`Scene`] into a visionOS immersive space.
///
/// Owns the frame loop: it takes the compositor's layer renderer, the device it
/// names, an ARKit world tracker, and a [`crate::metal::visionos::frame_loop::MetalRenderer`], and turns each
/// compositor frame into one stereo draw.
pub struct ImmersiveRenderer {
    layer: CpLayerRenderer,
    device: MetalDevice,
    renderer: MetalRenderer,
    tracking: Option<WorldTracking>,
    convention: AxisConvention,
    near: f32,
    far: f32,
    origin: Matrix4,
    last_stats: MetalRenderStats,
}

impl ImmersiveRenderer {
    /// Wrap the `cp_layer_renderer_t` SwiftUI's `CompositorLayer` provides.
    ///
    /// The renderer shares the compositor's `MTLDevice` rather than creating
    /// one — the drawable's textures belong to that device and no other.
    ///
    /// # Safety
    /// `layer_renderer` must be a live `cp_layer_renderer_t` that outlives this
    /// renderer. Call from the thread that will drive the frame loop.
    pub unsafe fn new(layer_renderer: *mut c_void) -> Result<Self, MetalError> {
        if layer_renderer.is_null() {
            return Err(MetalError::Unsupported("null cp_layer_renderer_t".into()));
        }
        let mtl_device = unsafe { cp_layer_renderer_get_device(layer_renderer) };
        let device = unsafe { MetalDevice::adopt(mtl_device) }?;
        let renderer = MetalRenderer::with_device(device.clone())?;
        // Tracking is not fatal to lose: without it the scene is head-locked,
        // which still draws.
        let tracking = WorldTracking::new().ok();
        Ok(Self {
            layer: layer_renderer,
            device,
            renderer,
            tracking,
            convention: AxisConvention::RightUpBack,
            near: MIN_NEAR,
            far: 1000.0,
            origin: Matrix4::identity(),
            last_stats: MetalRenderStats::default(),
        })
    }

    /// The compositor's current state.
    pub fn state(&self) -> LayerRendererState {
        LayerRendererState::from_raw(unsafe { cp_layer_renderer_get_state(self.layer) })
    }

    /// Block until the layer leaves the paused state.
    pub fn wait_until_running(&self) {
        unsafe { cp_layer_renderer_wait_until_running(self.layer) }
    }

    /// How many views the layer is configured for — 2 for stereo, 1 when the
    /// person has monocular rendering on.
    pub fn view_count(&self) -> usize {
        unsafe {
            let properties = cp_layer_renderer_get_properties(self.layer);
            if properties.is_null() {
                return 2;
            }
            cp_layer_renderer_properties_get_view_count(properties)
        }
    }

    /// Near and far clipping distances in metres. The compositor is told these
    /// each frame and builds the projection from them.
    ///
    /// The near plane is kept at 10 cm or further: CompositorServices takes
    /// nothing nearer (`cp_drawable_set_depth_range` stops the app).
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.near = near.max(MIN_NEAR);
        self.far = far.max(self.near + 0.001);
    }

    /// The clipping distances currently in force, as `(near, far)` metres.
    pub fn depth_range(&self) -> (f32, f32) {
        (self.near, self.far)
    }

    /// The NDC axis convention asked of `cp_drawable_compute_projection`.
    /// Defaults to [`AxisConvention::RightUpBack`], which is Metal's.
    pub fn set_axis_convention(&mut self, convention: AxisConvention) {
        self.convention = convention;
    }

    /// Where the space ARKit tracks the head in (the floor under the
    /// viewer at the start) is in the scene's world: the eyes are placed
    /// through it. The identity by default.
    pub fn set_origin(&mut self, origin: Matrix4) {
        self.origin = origin;
    }

    /// Whether world tracking came up. `false` means the scene is head-locked —
    /// usually a missing `NSWorldSensingUsageDescription`.
    pub fn is_world_tracked(&self) -> bool {
        self.tracking.is_some()
    }

    /// Statistics from the most recent frame.
    pub fn stats(&self) -> MetalRenderStats {
        self.last_stats
    }

    /// The underlying renderer, for cache control.
    pub fn renderer_mut(&mut self) -> &mut MetalRenderer {
        &mut self.renderer
    }

    /// Draw one compositor frame.
    ///
    /// Returns `false` once the layer is invalidated — the immersive space has
    /// closed and the loop should end. Returns `true` after a frame is
    /// submitted, and also when a frame is skipped (paused, or no drawable),
    /// because neither is a reason to stop.
    ///
    /// The sequence is the compositor's, in its order: query a frame, predict
    /// its timing, wait for the optimal input time, update, take the drawable,
    /// ask ARKit where the head is at the predicted presentation time, encode
    /// both eyes, present.
    pub fn render_frame(&mut self, scene: &mut Scene) -> Result<bool, MetalError> {
        self.render_frame_with(scene, |_, _| {})
    }

    /// As [`render_frame`](Self::render_frame), calling `update` once the
    /// frame's head pose and eyes are known and before they are drawn: the
    /// place for an app's per-frame work (input, animation, a panel placed
    /// by the head), which then shows in this frame rather than the next.
    /// `update` is not called for a frame that is skipped.
    pub fn render_frame_with(
        &mut self,
        scene: &mut Scene,
        update: impl FnOnce(&mut Scene, &ImmersiveFrame),
    ) -> Result<bool, MetalError> {
        match self.state() {
            LayerRendererState::Invalidated => return Ok(false),
            LayerRendererState::Paused => {
                self.wait_until_running();
                return Ok(true);
            }
            LayerRendererState::Running => {}
        }

        let _pool = AutoreleasePool::new();
        unsafe {
            let frame = cp_layer_renderer_query_next_frame(self.layer);
            if frame.is_null() {
                return Ok(true);
            }

            let timing = cp_frame_predict_timing(frame);
            if !timing.is_null() {
                cp_time_wait_until(cp_frame_timing_get_optimal_input_time(timing));
            }

            cp_frame_start_update(frame);
            // Scene mutation belongs between start_update and end_update: the
            // compositor uses the window to schedule against input.
            cp_frame_end_update(frame);

            cp_frame_start_submission(frame);
            let drawable = cp_frame_query_drawable(frame);
            if drawable.is_null() {
                cp_frame_end_submission(frame);
                return Ok(true);
            }

            // Reverse-Z: x is far, y is near. Not a typo — see `drawable.h`.
            cp_drawable_set_depth_range(
                drawable,
                SimdFloat2 {
                    x: self.far,
                    y: self.near,
                },
            );

            let head = self.update_device_anchor(drawable);
            let origin_from_device = head.unwrap_or_else(Matrix4::identity);
            let views = self.views(drawable, &origin_from_device);
            let pass = self.attachments(drawable);
            update(scene, &ImmersiveFrame { head, views: &views, size: pass.size() });

            let stats = self.renderer.render_views(scene, &views, &pass)?;
            self.last_stats = stats;

            // The present is encoded on a command buffer of its own, which
            // sequences behind the render on the same queue.
            let command_buffer = self.device.command_buffer();
            if !command_buffer.is_null() {
                cp_drawable_encode_present(drawable, command_buffer);
                let _: () = msg0(command_buffer, sel!("commit"));
            }
            cp_frame_end_submission(frame);
        }
        Ok(true)
    }

    /// Ask ARKit for the head pose at this frame's presentation time and give
    /// the anchor to the compositor, which needs it to reproject the frame.
    unsafe fn update_device_anchor(&self, drawable: CpDrawable) -> Option<Matrix4> {
        let tracking = self.tracking.as_ref()?;
        let timing = unsafe { cp_drawable_get_frame_timing(drawable) };
        if timing.is_null() {
            return None;
        }
        let presentation =
            unsafe { cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(timing)) };
        let pose = tracking.device_pose(presentation)?;
        unsafe { cp_drawable_set_device_anchor(drawable, tracking.anchor_ptr()) };
        Some(pose)
    }

    /// One [`RenderView`] per eye.
    ///
    /// `cp_view_get_transform` is device-from-eye, so world-from-eye is
    /// `origin_from_device * device_from_eye`, and the view matrix — world into
    /// eye — is its inverse.
    unsafe fn views(&self, drawable: CpDrawable, origin_from_device: &Matrix4) -> Vec<RenderView> {
        let count = unsafe { cp_drawable_get_view_count(drawable) }.min(super::super::MAX_VIEWS);
        let mut views = Vec::with_capacity(count);
        for index in 0..count {
            let view = unsafe { cp_drawable_get_view(drawable, index) };
            if view.is_null() {
                continue;
            }
            let device_from_eye: Matrix4 = unsafe { cp_view_get_transform(view) }.into();
            let world_from_eye = self.origin.multiply(&origin_from_device.multiply(&device_from_eye));
            let projection: Matrix4 =
                unsafe { cp_drawable_compute_projection(drawable, self.convention as u8, index) }
                    .into();

            let mut render_view = RenderView::from_matrices(world_from_eye.invert(), projection);
            let map = unsafe { cp_view_get_view_texture_map(view) };
            if !map.is_null() {
                render_view.slice = unsafe { cp_view_texture_map_get_slice_index(map) } as u32;
                let viewport = unsafe { cp_view_texture_map_get_viewport(map) };
                render_view.viewport = Some(viewport);
                // A texture index of its own means the dedicated layout: each
                // eye has its own texture rather than a slice of one.
                let texture_index = unsafe { cp_view_texture_map_get_texture_index(map) };
                if texture_index != 0 {
                    render_view.color =
                        unsafe { cp_drawable_get_color_texture(drawable, texture_index) };
                    render_view.depth =
                        unsafe { cp_drawable_get_depth_texture(drawable, texture_index) };
                }
            }
            views.push(render_view);
        }
        views
    }

    /// The pass the eyes render into: texture 0 of the drawable, reverse-Z.
    unsafe fn attachments(&self, drawable: CpDrawable) -> PassAttachments {
        let color = unsafe { cp_drawable_get_color_texture(drawable, 0) };
        let depth = unsafe { cp_drawable_get_depth_texture(drawable, 0) };
        let (width, height) = unsafe { texture_size(color) };
        unsafe {
            PassAttachments::from_raw(
                color,
                NIL,
                depth,
                NIL,
                width,
                height,
                texture_pixel_format(color),
                texture_pixel_format(depth),
                1,
            )
        }
        .with_reverse_z(true)
    }

    /// The compositor's device.
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }
}

/// `MTLTexture.width` / `.height`.
unsafe fn texture_size(texture: Id) -> (u32, u32) {
    if texture.is_null() {
        return (0, 0);
    }
    let width: NSUInteger = unsafe { msg0(texture, sel!("width")) };
    let height: NSUInteger = unsafe { msg0(texture, sel!("height")) };
    (width as u32, height as u32)
}

/// `MTLTexture.pixelFormat`.
unsafe fn texture_pixel_format(texture: Id) -> NSUInteger {
    if texture.is_null() {
        return 0;
    }
    unsafe { msg0(texture, sel!("pixelFormat")) }
}
