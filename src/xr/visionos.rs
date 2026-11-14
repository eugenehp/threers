//! visionOS, natively: an immersive space as an [`XrSource`], drawn by the
//! wgpu renderer (its Metal backend) into the compositor's own textures. So
//! a scene ([`Renderer::render_xr`](crate::renderer::Renderer::render_xr)),
//! a UI on a [`Panel`](super::Panel), or both
//! ([`PanelPass::draw_in_scene`](super::PanelPass::draw_in_scene)) are
//! drawn here exactly as on the web.
//!
//! The app is a SwiftUI `ImmersiveSpace` with a `CompositorLayer` whose
//! configuration asks for the **shared** layout (both eyes in one texture,
//! a viewport each: what `XrSource` describes). It hands the layer renderer
//! to Rust on its render thread, and forwards look-and-pinch:
//!
//! ```swift
//! CompositorLayer(configuration: SharedLayout()) { layerRenderer in
//!     layerRenderer.onSpatialEvent = { events in
//!         for e in events { /* threers_xr_spatial_event(id, phase, ray…) */ }
//!     }
//!     Thread { app_run(Unmanaged.passUnretained(layerRenderer).toOpaque()) }.start()
//! }
//! ```
//!
//! ```no_run
//! # fn f(layer: *mut std::ffi::c_void) {
//! use threers::xr::visionos::{Frame, ImmersiveSession};
//! let mut session = unsafe { ImmersiveSession::new(layer) }.unwrap();
//! loop {
//!     match session.begin_frame() {
//!         Frame::Draw(target) => {
//!             // draw each of session.eyes(..) into its viewport of `target`
//!             session.end_frame();
//!         }
//!         Frame::Skip => {}
//!         Frame::Ended => break,
//!     }
//! }
//! # }
//! ```
//!
//! A frame follows CompositorServices' pacing: wait for the optimal input
//! time, read the head (ARKit world tracking, at the predicted
//! presentation time) and the eyes, draw, then present on wgpu's own Metal
//! queue, behind the drawing. The compositor's depth (reverse-Z, which is
//! all it takes) is cleared to far: the renderer keeps its own depth, in
//! its own convention.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLPixelFormat, MTLTexture};

use super::{
    Handedness, InputSlots, TargetRayMode, XrEvent, XrEventKind, XrEye, XrInput, XrSource,
};
use crate::math::{Matrix4, Vector3};
use crate::metal::visionos::frame_loop::*;
use crate::metal::visionos::{
    AxisConvention, CpDrawable, CpFrame, CpLayerRenderer, LayerRendererState, SimdFloat2,
};

/// A look-and-pinch from the app (`SpatialEventCollection.Event`): which
/// one, its phase (0 began, 1 active — a start the first time an id is
/// seen —, 2 ended, 3 cancelled), and its selection ray in the immersive
/// space.
#[derive(Debug, Clone, Copy)]
struct Spatial {
    id: u64,
    phase: u32,
    origin: [f32; 3],
    direction: [f32; 3],
}

/// Pinches from the main thread, waiting for the render thread's next frame.
static SPATIAL: Mutex<Vec<Spatial>> = Mutex::new(Vec::new());

/// A harness's fingertip (see [`set_scripted_fingertip`]).
static SCRIPTED_TIP: Mutex<Option<[f32; 3]>> = Mutex::new(None);

/// For a harness (the simulator has no hands): a right hand whose joints
/// are all at `tip`, in the space's coordinates, as if tracked, from the
/// next frame; `None` takes it away. Its index fingertip pokes as a real
/// one does. Any thread, like [`threers_xr_spatial_event`].
pub fn set_scripted_fingertip(tip: Option<[f32; 3]>) {
    if let Ok(mut t) = SCRIPTED_TIP.lock() {
        *t = tip;
    }
}

/// visionOS 26's tracking areas (the system's highlight of what is looked
/// at) and drawable arrays, looked up when first needed, so an app built
/// for an older system still runs, without them.
struct Newer {
    tracking_texture_count: Option<unsafe extern "C" fn(CpDrawable) -> usize>,
    tracking_texture: Option<unsafe extern "C" fn(CpDrawable, usize) -> *mut c_void>,
    add_tracking_area: Option<unsafe extern "C" fn(CpDrawable, u64) -> *mut c_void>,
    render_value: Option<unsafe extern "C" fn(*mut c_void) -> u16>,
    add_hover_effect: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    query_drawables: Option<unsafe extern "C" fn(CpFrame) -> *mut c_void>,
    drawable_count: Option<unsafe extern "C" fn(*mut c_void) -> usize>,
    drawable_at: Option<unsafe extern "C" fn(*mut c_void, usize) -> CpDrawable>,
    drawable_target: Option<unsafe extern "C" fn(CpDrawable) -> u32>,
}

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
}

#[link(name = "CompositorServices", kind = "framework")]
unsafe extern "C" {
    fn cp_layer_renderer_set_minimum_frame_repeat_count(
        layer_renderer: CpLayerRenderer,
        count: i32,
    );
}

impl Newer {
    fn get() -> &'static Newer {
        static NEWER: std::sync::OnceLock<Newer> = std::sync::OnceLock::new();
        NEWER.get_or_init(|| {
            // RTLD_DEFAULT: every image loaded.
            let find =
                |name: &std::ffi::CStr| unsafe { dlsym(-2isize as *mut c_void, name.as_ptr()) };
            macro_rules! f {
                ($name:literal) => {{
                    let p = find($name);
                    // SAFETY: the symbol is the C function the field's type
                    // declares (CompositorServices' headers).
                    (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, _>(p) })
                }};
            }
            Newer {
                tracking_texture_count: f!(c"cp_drawable_get_tracking_areas_texture_count"),
                tracking_texture: f!(c"cp_drawable_get_tracking_areas_texture"),
                add_tracking_area: f!(c"cp_drawable_add_tracking_area"),
                render_value: f!(c"cp_tracking_area_get_render_value"),
                add_hover_effect: f!(c"cp_tracking_area_add_automatic_hover_effect"),
                query_drawables: f!(c"cp_frame_query_drawables"),
                drawable_count: f!(c"cp_drawable_array_get_count"),
                drawable_at: f!(c"cp_drawable_array_get_drawable"),
                drawable_target: f!(c"cp_drawable_get_target"),
            }
        })
    }
}

/// The system's accessibility settings, as the app's shell last reported
/// them (see [`threers_xr_set_accessibility`]): what a UI should follow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemAccessibility {
    pub reduce_motion: bool,
    pub bold_text: bool,
    pub increase_contrast: bool,
    pub reduce_transparency: bool,
    /// VoiceOver is on.
    pub voice_over: bool,
    /// The system's appearance is dark.
    pub dark: bool,
    /// The text size chosen, relative to the default (1).
    pub text_scale: f32,
}

impl Default for SystemAccessibility {
    fn default() -> Self {
        SystemAccessibility {
            reduce_motion: false,
            bold_text: false,
            increase_contrast: false,
            reduce_transparency: false,
            voice_over: false,
            dark: false,
            text_scale: 1.0,
        }
    }
}

static ACCESSIBILITY: Mutex<Option<SystemAccessibility>> = Mutex::new(None);

/// From the app's shell, at launch and whenever one changes: the system's
/// accessibility settings, `flags` a bit each (1 reduce motion, 2 bold
/// text, 4 increase contrast, 8 reduce transparency, 16 VoiceOver, 32 a
/// dark appearance), and the
/// text size relative to the default. Any thread.
#[unsafe(no_mangle)]
pub extern "C" fn threers_xr_set_accessibility(flags: u32, text_scale: f32) {
    let a = SystemAccessibility {
        reduce_motion: flags & 1 != 0,
        bold_text: flags & 2 != 0,
        increase_contrast: flags & 4 != 0,
        reduce_transparency: flags & 8 != 0,
        voice_over: flags & 16 != 0,
        dark: flags & 32 != 0,
        text_scale: if text_scale.is_finite() && text_scale > 0.0 {
            text_scale.clamp(0.5, 4.0)
        } else {
            1.0
        },
    };
    if let Ok(mut s) = ACCESSIBILITY.lock() {
        *s = Some(a);
    }
}

/// The system's accessibility settings, `None` until the shell has said
/// (an older shell never does).
pub fn system_accessibility() -> Option<SystemAccessibility> {
    ACCESSIBILITY.lock().ok().and_then(|s| *s)
}

/// How the shell reads a message out: the message (UTF-8, NUL-terminated)
/// and whether it interrupts.
type Announcer = extern "C" fn(*const std::ffi::c_char, bool);
static ANNOUNCER: Mutex<Option<Announcer>> = Mutex::new(None);

/// From the app's shell: how to have VoiceOver read a message out
/// (`AccessibilityNotification.Announcement`), for [`announce`]. Any
/// thread.
#[unsafe(no_mangle)]
pub extern "C" fn threers_xr_set_announcer(announcer: Option<Announcer>) {
    if let Ok(mut a) = ANNOUNCER.lock() {
        *a = announcer;
    }
}

/// Has VoiceOver read `message` out (`assertive`: at once), through the
/// shell's announcer; nothing if there is none. Returns whether there was.
pub fn announce(message: &str, assertive: bool) -> bool {
    let Some(f) = ANNOUNCER.lock().ok().and_then(|a| *a) else {
        return false;
    };
    let Ok(text) = std::ffi::CString::new(message.replace('\0', " ")) else {
        return false;
    };
    f(text.as_ptr(), assertive);
    true
}

/// From the app's `onSpatialEvent`: a pinch `id` began (`phase` 0), moved
/// (1), ended (2) or was cancelled (3), with its selection ray (`origin`,
/// `direction`) in the immersive space. Any thread.
#[unsafe(no_mangle)]
pub extern "C" fn threers_xr_spatial_event(
    id: u64,
    phase: u32,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) {
    if let Ok(mut q) = SPATIAL.lock() {
        q.push(Spatial {
            id,
            phase,
            origin: [ox, oy, oz],
            direction: [dx, dy, dz],
        });
    }
}

#[link(name = "ARKit", kind = "framework")]
unsafe extern "C" {
    fn ar_hand_tracking_provider_is_supported() -> bool;
    fn ar_hand_tracking_configuration_create() -> *mut c_void;
    fn ar_hand_tracking_provider_create(configuration: *mut c_void) -> *mut c_void;
    fn ar_hand_anchor_create() -> *mut c_void;
    fn ar_hand_tracking_provider_get_latest_anchors(
        provider: *mut c_void,
        left: *mut c_void,
        right: *mut c_void,
    ) -> bool;
    fn ar_hand_anchor_is_tracked(anchor: *mut c_void) -> bool;
    fn ar_hand_anchor_get_hand_skeleton(anchor: *mut c_void) -> *mut c_void;
    fn ar_hand_skeleton_get_joint_named(skeleton: *mut c_void, name: u64) -> *mut c_void;
    // `simd_float4x4` returns: called through `call_returning_float4x4`.
    #[link_name = "ar_hand_anchor_get_origin_from_anchor_transform"]
    fn ar_hand_anchor_get_origin_from_anchor_transform_sym();
    #[link_name = "ar_skeleton_joint_get_anchor_from_joint_transform"]
    fn ar_skeleton_joint_get_anchor_from_joint_transform_sym();
    #[link_name = "ar_release"]
    fn ar_release_hand(object: *mut c_void);
}

/// The hands' input ids (apart from pinches').
const HAND_LEFT: u64 = 1 << 40;
const HAND_RIGHT: u64 = (1 << 40) + 1;

/// About how thick each joint is (ARKit gives no radii): fingertips and
/// knuckles a little under a centimetre, the wrist more.
const HAND_RADII: [f32; 25] = [
    0.02, 0.012, 0.01, 0.009, 0.008, 0.011, 0.01, 0.009, 0.008, 0.007, 0.011, 0.01, 0.009, 0.008,
    0.007, 0.011, 0.01, 0.009, 0.008, 0.007, 0.011, 0.009, 0.008, 0.007, 0.006,
];

/// ARKit's hand tracking: a provider run with world tracking, and the two
/// anchors it fills each frame.
struct HandTracking {
    provider: *mut c_void,
    left: *mut c_void,
    right: *mut c_void,
}

impl HandTracking {
    /// `None` where the device has no hand tracking (the simulator).
    fn new() -> Option<Self> {
        unsafe {
            if !ar_hand_tracking_provider_is_supported() {
                return None;
            }
            let configuration = ar_hand_tracking_configuration_create();
            let provider = ar_hand_tracking_provider_create(configuration);
            ar_release_hand(configuration);
            (!provider.is_null()).then(|| HandTracking {
                provider,
                left: ar_hand_anchor_create(),
                right: ar_hand_anchor_create(),
            })
        }
    }

    /// The tracked hands now: id, which hand, and the poses of the 25
    /// joints WebXR names (ARKit's first 25, in the same order), in the
    /// space's coordinates.
    unsafe fn latest(&self) -> Vec<(u64, Handedness, Vec<f32>)> {
        let mut out = vec![];
        if !unsafe {
            ar_hand_tracking_provider_get_latest_anchors(self.provider, self.left, self.right)
        } {
            return out;
        }
        for (anchor, id, handed) in [
            (self.left, HAND_LEFT, Handedness::Left),
            (self.right, HAND_RIGHT, Handedness::Right),
        ] {
            unsafe {
                if !ar_hand_anchor_is_tracked(anchor) {
                    continue;
                }
                let skeleton = ar_hand_anchor_get_hand_skeleton(anchor);
                if skeleton.is_null() {
                    continue;
                }
                let origin: Matrix4 = call_returning_float4x4(
                    ar_hand_anchor_get_origin_from_anchor_transform_sym,
                    anchor as usize,
                    0,
                    0,
                )
                .into();
                let mut joints = Vec::with_capacity(25 * 16);
                for k in 0..25u64 {
                    let joint = ar_hand_skeleton_get_joint_named(skeleton, k);
                    let local: Matrix4 = call_returning_float4x4(
                        ar_skeleton_joint_get_anchor_from_joint_transform_sym,
                        joint as usize,
                        0,
                        0,
                    )
                    .into();
                    joints.extend_from_slice(&origin.multiply(&local).elements);
                }
                out.push((id, handed, joints));
            }
        }
        out
    }
}

impl Drop for HandTracking {
    fn drop(&mut self) {
        unsafe {
            ar_release_hand(self.left);
            ar_release_hand(self.right);
            ar_release_hand(self.provider);
        }
    }
}

/// What [`ImmersiveSession::begin_frame`] found.
pub enum Frame {
    /// Draw this frame's eyes into the target, then
    /// [`end_frame`](ImmersiveSession::end_frame).
    Draw(wgpu::TextureView),
    /// No frame now (paused, or none ready); ask again.
    Skip,
    /// The space was closed: stop.
    Ended,
}

/// One eye of the frame: its pose in the space, projection (0..1 depth)
/// and viewport (from the bottom left).
#[derive(Debug, Clone, Copy)]
struct Eye {
    pose: Matrix4,
    projection: [f32; 16],
    viewport: [i32; 4],
}

/// The frame being drawn.
struct Current {
    frame: CpFrame,
    drawable: CpDrawable,
    color: wgpu::Texture,
    depth: Option<wgpu::Texture>,
    /// The tracking areas' texture, where the shell asked for one.
    hover: Option<wgpu::Texture>,
}

/// A visionOS immersive space: its frames, the head, and look-and-pinch, as
/// an [`XrSource`]. See the module docs.
pub struct ImmersiveSession {
    layer: CpLayerRenderer,
    tracking: Option<WorldTracking>,
    /// ARKit hand tracking, where there is any (not in the simulator).
    hands: Option<HandTracking>,
    /// This frame's tracked hands: id, which hand, 25 joint poses.
    hands_now: Vec<(u64, Handedness, Vec<f32>)>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    near: f32,
    far: f32,
    see_through: bool,
    current: Option<Current>,
    eyes: Vec<Eye>,
    head: Option<[f32; 16]>,
    /// The last pose world tracking gave, held while it is lost.
    tracked: Option<Matrix4>,
    /// Where the head is taken to be before tracking has given a pose (and
    /// where there is none: the simulator).
    untracked_head: Matrix4,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    slots: InputSlots<u64>,
    rays: HashMap<u64, [f32; 16]>,
    events: Vec<XrEvent>,
    /// Pinches ended this frame: reported once more, then let go.
    leaving: Vec<u64>,
    /// The tracking areas declared this frame, and their values.
    hover_areas: std::cell::RefCell<HashMap<u64, u32>>,
}

// The session lives on the render thread the app starts; the raw handles
// are CompositorServices' and only used there.
unsafe impl Send for ImmersiveSession {}

impl ImmersiveSession {
    /// The session for `layer_renderer` (a `cp_layer_renderer_t`, from the
    /// app's `CompositorLayer`): a wgpu device on Metal, and ARKit world
    /// tracking (needing `NSWorldSensingUsageDescription` in the app's
    /// Info.plist; without it the view is head-locked).
    ///
    /// # Safety
    /// `layer_renderer` must be a live `cp_layer_renderer_t`, used from the
    /// thread that calls this from then on.
    pub unsafe fn new(layer_renderer: *mut c_void) -> Result<Self, String> {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no Metal adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("threers visionOS"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("no device: {e}"))?;
        // Hand tracking where the device has it; the app's Info.plist needs
        // NSHandsTrackingUsageDescription, and the user's leave.
        let hands = HandTracking::new();
        Ok(ImmersiveSession {
            layer: layer_renderer,
            tracking: WorldTracking::with_providers(
                &hands.as_ref().map(|h| vec![h.provider]).unwrap_or_default(),
            )
            .ok(),
            hands,
            hands_now: vec![],
            device: Arc::new(device),
            queue: Arc::new(queue),
            // The compositor's minimum near plane.
            near: 0.1,
            far: 100.0,
            see_through: false,
            current: None,
            eyes: vec![],
            head: None,
            tracked: None,
            untracked_head: Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)),
            size: (1, 1),
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            slots: InputSlots::new(),
            rays: HashMap::new(),
            events: vec![],
            leaving: vec![],
            hover_areas: Default::default(),
        })
    }

    /// The device everything is drawn with (pipelines, textures, a
    /// [`Renderer`](crate::renderer::Renderer)).
    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    /// The compositor's colour format, as of the last frame (the layer's
    /// configuration chooses it; `Bgra8UnormSrgb` by default).
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The near and far planes the eyes' projections use from the next
    /// frame. The compositor takes no near plane under 0.1 m.
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        (self.near, self.far) = (near.max(0.1), far.max(near.max(0.1) + 0.01));
    }

    /// For a mixed or progressive space, where the room shows through:
    /// draw no background then.
    pub fn set_see_through(&mut self, on: bool) {
        self.see_through = on;
    }

    /// Shows each frame `count + 1` times: 0 draws at the display's rate
    /// (90 Hz), 1 at half. For a still scene, it halves the power used.
    pub fn set_frame_repeat(&self, count: u32) {
        unsafe {
            cp_layer_renderer_set_minimum_frame_repeat_count(self.layer, count.min(8) as i32)
        };
    }

    /// Waits for the next frame and readies it: the head, the eyes and the
    /// inputs; returns the texture to draw them into.
    pub fn begin_frame(&mut self) -> Frame {
        unsafe {
            match LayerRendererState::from_raw(cp_layer_renderer_get_state(self.layer)) {
                LayerRendererState::Invalidated => return Frame::Ended,
                LayerRendererState::Paused => {
                    cp_layer_renderer_wait_until_running(self.layer);
                    return Frame::Skip;
                }
                LayerRendererState::Running => {}
            }
            let frame = cp_layer_renderer_query_next_frame(self.layer);
            if frame.is_null() {
                return Frame::Skip;
            }
            let timing = cp_frame_predict_timing(frame);
            if !timing.is_null() {
                cp_time_wait_until(cp_frame_timing_get_optimal_input_time(timing));
            }
            cp_frame_start_update(frame);
            self.take_spatial();
            self.read_hands();
            cp_frame_end_update(frame);
            cp_frame_start_submission(frame);
            let drawable = Self::query_drawable(frame);
            if drawable.is_null() {
                cp_frame_end_submission(frame);
                return Frame::Skip;
            }
            // Reverse-Z: x is far, y near (see `drawable.h`).
            cp_drawable_set_depth_range(
                drawable,
                SimdFloat2 {
                    x: self.far,
                    y: self.near,
                },
            );
            let origin_from_device = self.device_anchor(drawable);
            self.head = Some(origin_from_device.elements);
            let color_raw = cp_drawable_get_color_texture(drawable, 0) as *mut c_void;
            let depth_raw = cp_drawable_get_depth_texture(drawable, 0) as *mut c_void;
            let Some(color) = self.wrap(color_raw, wgpu::TextureUsages::RENDER_ATTACHMENT) else {
                cp_frame_end_submission(frame);
                return Frame::Skip;
            };
            self.size = (color.width(), color.height());
            self.format = color.format();
            let depth = self.wrap(depth_raw, wgpu::TextureUsages::RENDER_ATTACHMENT);
            // Tracking areas: a texture of them where the shell asked for
            // one (`trackingAreasFormat`), and none declared yet.
            self.hover_areas.borrow_mut().clear();
            let newer = Newer::get();
            let hover = match (newer.tracking_texture_count, newer.tracking_texture) {
                (Some(count), Some(get)) if count(drawable) > 0 => {
                    self.wrap(get(drawable, 0), wgpu::TextureUsages::RENDER_ATTACHMENT)
                }
                _ => None,
            };
            self.eyes.clear();
            let count = cp_drawable_get_view_count(drawable);
            for i in 0..count {
                let view = cp_drawable_get_view(drawable, i);
                if view.is_null() {
                    continue;
                }
                let device_from_eye: Matrix4 = cp_view_get_transform(view).into();
                let reversed: Matrix4 =
                    cp_drawable_compute_projection(drawable, AxisConvention::RightUpBack as u8, i)
                        .into();
                // Near at 1 to near at 0: z' = w - z.
                let mut projection = reversed.elements;
                for c in 0..4 {
                    projection[c * 4 + 2] = projection[c * 4 + 3] - projection[c * 4 + 2];
                }
                let map = cp_view_get_view_texture_map(view);
                let viewport = if map.is_null() {
                    [0, 0, self.size.0 as i32, self.size.1 as i32]
                } else {
                    // Metal's rectangle is from the top; the eyes' from the bottom.
                    let v = cp_view_texture_map_get_viewport(map);
                    let (x, y, w, h) = (
                        v.origin_x as i32,
                        v.origin_y as i32,
                        v.width as i32,
                        v.height as i32,
                    );
                    [x, self.size.1 as i32 - y - h, w, h]
                };
                self.eyes.push(Eye {
                    pose: origin_from_device.multiply(&device_from_eye),
                    projection,
                    viewport,
                });
            }
            if std::env::var_os("THREERS_XR_DEBUG").is_some() {
                eprintln!(
                    "threers xr: {:?} {:?} head {:?} eyes {:?}",
                    self.size, self.format, self.head, self.eyes
                );
            }
            let target = color.create_view(&Default::default());
            self.current = Some(Current {
                frame,
                drawable,
                color,
                depth,
                hover,
            });
            Frame::Draw(target)
        }
    }

    /// Presents what was drawn (behind it, on the same queue) and closes the
    /// frame.
    pub fn end_frame(&mut self) {
        let Some(cur) = self.current.take() else {
            return;
        };
        // The compositor's depth, cleared to far: nothing claims to be near.
        if let Some(depth) = &cur.depth {
            let view = depth.create_view(&Default::default());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers visionOS depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.queue.submit(Some(encoder.finish()));
        }
        unsafe {
            if let Some(hal) = self.queue.as_hal::<wgpu::hal::api::Metal>()
                && let Some(buffer) = hal.as_raw().commandBuffer()
            {
                cp_drawable_encode_present(
                    cur.drawable,
                    Retained::as_ptr(&buffer) as *mut c_void as _,
                );
                buffer.commit();
            }
            cp_frame_end_submission(cur.frame);
        }
        drop(cur.color);
    }

    /// The frame's drawable for the headset's own display. From visionOS 26
    /// a frame may have several (one for a recording or AirPlay too), and
    /// with tracking areas it must be asked for them all; a recording then
    /// gets nothing drawn.
    unsafe fn query_drawable(frame: CpFrame) -> CpDrawable {
        let newer = Newer::get();
        let (Some(all), Some(count), Some(get), Some(target)) = (
            newer.query_drawables,
            newer.drawable_count,
            newer.drawable_at,
            newer.drawable_target,
        ) else {
            return unsafe { cp_frame_query_drawable(frame) };
        };
        let array = unsafe { all(frame) };
        if array.is_null() {
            return std::ptr::null_mut();
        }
        let n = unsafe { count(array) };
        let drawables: Vec<CpDrawable> = (0..n)
            .map(|i| unsafe { get(array, i) })
            .filter(|d| !d.is_null())
            .collect();
        // 0: the built-in display.
        drawables
            .iter()
            .copied()
            .find(|&d| unsafe { target(d) } == 0)
            .or(drawables.first().copied())
            .unwrap_or(std::ptr::null_mut())
    }

    /// The head's pose at this frame's presentation, handed to the
    /// compositor too (it reprojects with it). Identity until tracking runs.
    unsafe fn device_anchor(&mut self, drawable: CpDrawable) -> Matrix4 {
        let fallback = self.tracked.unwrap_or(self.untracked_head);
        let Some(tracking) = &self.tracking else {
            return fallback;
        };
        let timing = unsafe { cp_drawable_get_frame_timing(drawable) };
        if timing.is_null() {
            return fallback;
        }
        let at =
            unsafe { cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(timing)) };
        match tracking.device_pose(at) {
            // The simulator's tracking "succeeds" with the space's origin
            // exactly — the head on the floor. A real pose is never exactly
            // that (it is a metre and more up, and noisy), so it counts as
            // no pose.
            Some(pose) if pose != Matrix4::identity() => {
                unsafe { cp_drawable_set_device_anchor(drawable, tracking.anchor_ptr()) };
                self.tracked = Some(pose);
                pose
            }
            // Lost, or not yet found: the last pose, or a standing head.
            _ => fallback,
        }
    }

    /// Where the head is taken to be until world tracking gives a pose (in
    /// the simulator, always): standing, 1.6 m up, by default — rather than
    /// at the space's origin, which is on the floor.
    pub fn set_untracked_head(&mut self, pose: Matrix4) {
        self.untracked_head = pose;
    }

    /// Whether world tracking has given a pose yet.
    pub fn is_tracked(&self) -> bool {
        self.tracked.is_some()
    }

    /// A compositor texture as a wgpu one (it stays the compositor's).
    unsafe fn wrap(&self, raw: *mut c_void, usage: wgpu::TextureUsages) -> Option<wgpu::Texture> {
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            unsafe { Retained::retain(raw.cast()) }?;
        let format = match texture.pixelFormat() {
            MTLPixelFormat::BGRA8Unorm_sRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
            MTLPixelFormat::BGRA8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            MTLPixelFormat::RGBA8Unorm_sRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
            MTLPixelFormat::RGBA8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            MTLPixelFormat::RGBA16Float => wgpu::TextureFormat::Rgba16Float,
            MTLPixelFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
            MTLPixelFormat::R8Uint => wgpu::TextureFormat::R8Uint,
            MTLPixelFormat::R16Uint => wgpu::TextureFormat::R16Uint,
            _ => return None,
        };
        let (w, h, layers) = (
            texture.width() as u32,
            texture.height() as u32,
            (texture.arrayLength() as u32).max(1),
        );
        let kind = texture.textureType();
        let hal = unsafe {
            wgpu::hal::metal::Device::texture_from_raw(
                texture,
                format,
                kind,
                layers,
                1,
                wgpu::hal::CopyExtent {
                    width: w,
                    height: h,
                    depth: 1,
                },
                None,
            )
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("threers visionOS drawable"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };
        Some(unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Metal>(
                    hal,
                    &desc,
                    wgpu::TextureUses::UNINITIALIZED,
                )
        })
    }

    /// The hands ARKit has now: each tracked one connected as an input
    /// with its joints, each lost one disconnected.
    fn read_hands(&mut self) {
        let mut now = match &self.hands {
            Some(h) => unsafe { h.latest() },
            None => vec![],
        };
        // A harness's right hand, its index fingertip where it was put.
        if let Some([x, y, z]) = SCRIPTED_TIP.lock().ok().and_then(|t| *t) {
            now.retain(|h| h.0 != HAND_RIGHT);
            let joints = Matrix4::translation(Vector3::new(x, y, z))
                .elements
                .repeat(25);
            now.push((HAND_RIGHT, Handedness::Right, joints));
        }
        for (id, _) in [(HAND_LEFT, ()), (HAND_RIGHT, ())] {
            let tracked = now.iter().any(|x| x.0 == id);
            match (tracked, self.slots.slot_of(&id)) {
                (true, None) => {
                    let slot = self.slots.connect(id);
                    self.events.push(XrEvent {
                        kind: XrEventKind::Connected,
                        slot,
                    });
                }
                (false, Some(slot)) => {
                    self.slots.disconnect(&id);
                    self.events.push(XrEvent {
                        kind: XrEventKind::Disconnected,
                        slot,
                    });
                }
                _ => {}
            }
        }
        self.hands_now = now;
    }

    /// The pinches the app forwarded since the last frame, as inputs that
    /// exist while they last (WebXR's `transient-pointer`).
    fn take_spatial(&mut self) {
        for id in std::mem::take(&mut self.leaving) {
            if let Some(slot) = self.slots.disconnect(&id) {
                self.events.push(XrEvent {
                    kind: XrEventKind::Disconnected,
                    slot,
                });
            }
            self.rays.remove(&id);
        }
        let events = SPATIAL
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default();
        for e in events {
            self.rays.insert(
                e.id,
                super::ray_pose(
                    Vector3::new(e.origin[0], e.origin[1], e.origin[2]),
                    Vector3::new(e.direction[0], e.direction[1], e.direction[2]),
                ),
            );
            // SwiftUI says only active, ended or cancelled: an id's first
            // active is its start.
            let starts = e.phase == 0 || (e.phase == 1 && self.slots.slot_of(&e.id).is_none());
            match e.phase {
                _ if starts => {
                    let slot = self.slots.connect(e.id);
                    self.events.push(XrEvent {
                        kind: XrEventKind::Connected,
                        slot,
                    });
                    self.events.push(XrEvent {
                        kind: XrEventKind::SelectStart,
                        slot,
                    });
                }
                2 | 3 => {
                    if let Some(slot) = self.slots.slot_of(&e.id) {
                        self.events.push(XrEvent {
                            kind: XrEventKind::SelectEnd,
                            slot,
                        });
                        if e.phase == 2 {
                            self.events.push(XrEvent {
                                kind: XrEventKind::Select,
                                slot,
                            });
                        }
                    }
                    self.leaving.push(e.id);
                }
                _ => {}
            }
        }
    }
}

impl XrSource for ImmersiveSession {
    fn head(&self) -> Option<[f32; 16]> {
        self.head
    }
    fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
        self.eyes
            .iter()
            .map(|e| XrEye {
                projection: e.projection,
                world: origin.multiply(&e.pose),
                viewport: e.viewport,
            })
            .collect()
    }
    fn framebuffer_size(&self) -> (u32, u32) {
        self.size
    }
    fn inputs(&self) -> Vec<XrInput> {
        self.slots
            .iter()
            .map(
                |(slot, id)| match self.hands_now.iter().find(|h| h.0 == *id) {
                    // A hand: its joints, no ray (its pinches come as spatial events).
                    Some((_, handedness, joints)) => XrInput {
                        slot,
                        handedness: *handedness,
                        joints: Some((joints.clone(), HAND_RADII.to_vec())),
                        ..Default::default()
                    },
                    None => XrInput {
                        slot,
                        target_ray_mode: TargetRayMode::TransientPointer,
                        ray: self.rays.get(id).copied(),
                        ..Default::default()
                    },
                },
            )
            .collect()
    }
    fn take_events(&mut self) -> Vec<XrEvent> {
        std::mem::take(&mut self.events)
    }
    fn see_through(&self) -> bool {
        self.see_through
    }
    fn hover_target(&self) -> Option<super::HoverTarget> {
        let t = self.current.as_ref()?.hover.as_ref()?;
        Some(super::HoverTarget {
            view: t.create_view(&Default::default()),
            format: t.format(),
            size: (t.width(), t.height()),
        })
    }
    fn declare_hover(&self, id: u64) -> Option<u32> {
        let cur = self.current.as_ref()?;
        cur.hover.as_ref()?;
        if id == 0 {
            return None;
        }
        if let Some(&v) = self.hover_areas.borrow().get(&id) {
            return Some(v);
        }
        let newer = Newer::get();
        let (add, value) = (newer.add_tracking_area?, newer.render_value?);
        let area = unsafe { add(cur.drawable, id) };
        if area.is_null() {
            return None;
        }
        if let Some(effect) = newer.add_hover_effect {
            unsafe { effect(area) };
        }
        // 0 is no area: none left this frame.
        let v = unsafe { value(area) } as u32;
        (v != 0).then(|| {
            self.hover_areas.borrow_mut().insert(id, v);
            v
        })
    }
}

/// Runs `A` in the immersive space whose layer renderer is `layer_renderer`
/// (a `cp_layer_renderer_t`), until the space closes: the scene built by
/// [`App::setup`](super::App::setup), changed by
/// [`App::update`](super::App::update) each frame, drawn into both eyes by
/// [`App::draw`](super::App::draw). Call it on a thread of its own (it waits for each
/// frame) — the SwiftUI shell does; [`visionos_app!`](crate::visionos_app)
/// exports the function it calls.
///
/// # Safety
/// `layer_renderer` must be the `CompositorLayer`'s live layer renderer.
pub unsafe fn run<A: super::App>(layer_renderer: *mut c_void) {
    let mut session = match unsafe { ImmersiveSession::new(layer_renderer) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("threers: no immersive session: {e}");
            return;
        }
    };
    let mut scene = crate::scene::Scene::new();
    let mut app = A::setup(&mut scene);
    let lens = app.lens();
    session.set_depth_range(lens.near, lens.far);
    let (device, queue) = (session.device(), session.queue());
    let mut renderer: Option<crate::renderer::Renderer> = None;
    let start = std::time::Instant::now();
    let mut last = 0.0;
    loop {
        let target = match session.begin_frame() {
            Frame::Draw(target) => target,
            Frame::Skip => continue,
            Frame::Ended => break,
        };
        let time = start.elapsed().as_secs_f64();
        let frame = super::Frame::take(&mut session, time, time - last);
        last = time;
        app.update(&mut scene, &frame);
        if frame.see_through {
            scene.background_alpha = 0.0;
        }
        let (w, h) = session.framebuffer_size();
        let format = session.format();
        let renderer = renderer.get_or_insert_with(|| {
            crate::renderer::Renderer::new_lazy(device.clone(), queue.clone(), format, w, h)
        });
        let mut gpu = super::Gpu {
            device: &device,
            queue: &queue,
            format,
            target: &target,
            renderer,
            source: &session,
        };
        app.draw(&mut scene, &frame, &mut gpu);
        session.end_frame();
    }
}
