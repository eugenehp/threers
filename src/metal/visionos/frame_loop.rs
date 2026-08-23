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

#[link(name = "CompositorServices", kind = "framework")]
extern "C" {
    fn cp_layer_renderer_get_state(layer_renderer: CpLayerRenderer) -> u32;
    fn cp_layer_renderer_wait_until_running(layer_renderer: CpLayerRenderer);
    fn cp_layer_renderer_query_next_frame(layer_renderer: CpLayerRenderer) -> CpFrame;
    fn cp_layer_renderer_get_device(layer_renderer: CpLayerRenderer) -> Id;
    fn cp_layer_renderer_get_properties(
        layer_renderer: CpLayerRenderer,
    ) -> CpLayerRendererProperties;
    fn cp_layer_renderer_properties_get_view_count(
        properties: CpLayerRendererProperties,
    ) -> NSUInteger;

    fn cp_frame_predict_timing(frame: CpFrame) -> CpFrameTiming;
    fn cp_frame_start_update(frame: CpFrame);
    fn cp_frame_end_update(frame: CpFrame);
    fn cp_frame_start_submission(frame: CpFrame);
    fn cp_frame_end_submission(frame: CpFrame);
    fn cp_frame_query_drawable(frame: CpFrame) -> CpDrawable;

    fn cp_frame_timing_get_optimal_input_time(timing: CpFrameTiming) -> CpTime;
    fn cp_frame_timing_get_presentation_time(timing: CpFrameTiming) -> CpTime;
    fn cp_time_wait_until(time: CpTime);
    fn cp_time_to_cf_time_interval(time: CpTime) -> f64;

    fn cp_drawable_get_view_count(drawable: CpDrawable) -> NSUInteger;
    fn cp_drawable_get_view(drawable: CpDrawable, index: NSUInteger) -> CpView;
    fn cp_drawable_get_color_texture(drawable: CpDrawable, index: NSUInteger) -> Id;
    fn cp_drawable_get_depth_texture(drawable: CpDrawable, index: NSUInteger) -> Id;
    fn cp_drawable_get_frame_timing(drawable: CpDrawable) -> CpFrameTiming;
    fn cp_drawable_set_device_anchor(drawable: CpDrawable, device_anchor: ArDeviceAnchor);
    fn cp_drawable_set_depth_range(drawable: CpDrawable, depth_range: SimdFloat2);
    fn cp_drawable_compute_projection(
        drawable: CpDrawable,
        convention: u8,
        view_index: NSUInteger,
    ) -> SimdFloat4x4;
    fn cp_drawable_encode_present(drawable: CpDrawable, command_buffer: Id);

    fn cp_view_get_transform(view: CpView) -> SimdFloat4x4;
    fn cp_view_get_view_texture_map(view: CpView) -> CpViewTextureMap;
    fn cp_view_texture_map_get_texture_index(map: CpViewTextureMap) -> NSUInteger;
    fn cp_view_texture_map_get_slice_index(map: CpViewTextureMap) -> NSUInteger;
    fn cp_view_texture_map_get_viewport(map: CpViewTextureMap) -> MTLViewport;
}

#[link(name = "ARKit", kind = "framework")]
extern "C" {
    fn ar_session_create() -> ArSession;
    fn ar_session_run(session: ArSession, data_providers: ArDataProviders);
    fn ar_world_tracking_configuration_create() -> ArWorldTrackingConfiguration;
    fn ar_world_tracking_provider_create(
        configuration: ArWorldTrackingConfiguration,
    ) -> ArWorldTrackingProvider;
    fn ar_world_tracking_provider_query_device_anchor_at_timestamp(
        provider: ArWorldTrackingProvider,
        timestamp: f64,
        device_anchor: ArDeviceAnchor,
    ) -> isize;
    fn ar_data_providers_create() -> ArDataProviders;
    fn ar_data_providers_add_data_provider(providers: ArDataProviders, provider: *mut c_void);
    fn ar_device_anchor_create() -> ArDeviceAnchor;
    fn ar_device_anchor_get_origin_from_anchor_transform(anchor: ArDeviceAnchor) -> SimdFloat4x4;
    fn ar_release(object: *mut c_void);
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
        unsafe {
            let configuration = ArObject::new(ar_world_tracking_configuration_create())
                .ok_or_else(|| MetalError::Unsupported("ar_world_tracking_configuration".into()))?;
            let provider = ArObject::new(ar_world_tracking_provider_create(configuration.ptr()))
                .ok_or_else(|| MetalError::Unsupported("ar_world_tracking_provider".into()))?;
            let providers = ArObject::new(ar_data_providers_create())
                .ok_or_else(|| MetalError::Unsupported("ar_data_providers".into()))?;
            ar_data_providers_add_data_provider(providers.ptr(), provider.ptr());

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
    fn anchor_ptr(&self) -> ArDeviceAnchor {
        self.anchor.ptr()
    }
}

// ------------------------------------------------------------------ renderer

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
        let mtl_device = cp_layer_renderer_get_device(layer_renderer);
        let device = MetalDevice::adopt(mtl_device)?;
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
            near: 0.01,
            far: 1000.0,
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
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.near = near.max(0.001);
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

            let origin_from_device = self.update_device_anchor(drawable);
            let views = self.views(drawable, &origin_from_device);
            let pass = self.attachments(drawable);

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
    unsafe fn update_device_anchor(&self, drawable: CpDrawable) -> Matrix4 {
        let Some(tracking) = &self.tracking else {
            return Matrix4::identity();
        };
        let timing = cp_drawable_get_frame_timing(drawable);
        if timing.is_null() {
            return Matrix4::identity();
        }
        let presentation =
            cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(timing));
        match tracking.device_pose(presentation) {
            Some(pose) => {
                cp_drawable_set_device_anchor(drawable, tracking.anchor_ptr());
                pose
            }
            None => Matrix4::identity(),
        }
    }

    /// One [`RenderView`] per eye.
    ///
    /// `cp_view_get_transform` is device-from-eye, so world-from-eye is
    /// `origin_from_device * device_from_eye`, and the view matrix — world into
    /// eye — is its inverse.
    unsafe fn views(&self, drawable: CpDrawable, origin_from_device: &Matrix4) -> Vec<RenderView> {
        let count = cp_drawable_get_view_count(drawable).min(super::super::MAX_VIEWS);
        let mut views = Vec::with_capacity(count);
        for index in 0..count {
            let view = cp_drawable_get_view(drawable, index);
            if view.is_null() {
                continue;
            }
            let device_from_eye: Matrix4 = cp_view_get_transform(view).into();
            let world_from_eye = origin_from_device.multiply(&device_from_eye);
            let projection: Matrix4 =
                cp_drawable_compute_projection(drawable, self.convention as u8, index).into();

            let mut render_view = RenderView::from_matrices(world_from_eye.invert(), projection);
            let map = cp_view_get_view_texture_map(view);
            if !map.is_null() {
                render_view.slice = cp_view_texture_map_get_slice_index(map) as u32;
                let viewport = cp_view_texture_map_get_viewport(map);
                render_view.viewport = Some(viewport);
                // A texture index of its own means the dedicated layout: each
                // eye has its own texture rather than a slice of one.
                let texture_index = cp_view_texture_map_get_texture_index(map);
                if texture_index != 0 {
                    render_view.color = cp_drawable_get_color_texture(drawable, texture_index);
                    render_view.depth = cp_drawable_get_depth_texture(drawable, texture_index);
                }
            }
            views.push(render_view);
        }
        views
    }

    /// The pass the eyes render into: texture 0 of the drawable, reverse-Z.
    unsafe fn attachments(&self, drawable: CpDrawable) -> PassAttachments {
        let color = cp_drawable_get_color_texture(drawable, 0);
        let depth = cp_drawable_get_depth_texture(drawable, 0);
        let (width, height) = texture_size(color);
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
    let width: NSUInteger = msg0(texture, sel!("width"));
    let height: NSUInteger = msg0(texture, sel!("height"));
    (width as u32, height as u32)
}

/// `MTLTexture.pixelFormat`.
unsafe fn texture_pixel_format(texture: Id) -> NSUInteger {
    if texture.is_null() {
        return 0;
    }
    msg0(texture, sel!("pixelFormat"))
}
