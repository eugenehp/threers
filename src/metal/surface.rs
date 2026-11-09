//! Presenting to a window, via `CAMetalLayer`.
//!
//! The crate does not open the window — winit (or AppKit, UIKit, SDL) does
//! that, and hands over the view or its layer; this attaches a Metal layer to
//! it and hands back a drawable per frame.

use std::ffi::c_void;

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg1, sel, AutoreleasePool, Bool, Id, Owned, NIL, YES};
use super::renderer::PassAttachments;
use super::resources::new_texture_2d;

// CAMetalLayer lives in QuartzCore. The empty extern block is enough to make
// rustc pass `-framework QuartzCore` to the linker, which is what registers the
// class with the Objective-C runtime.
#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {}

/// A `CAMetalLayer` attached to a view, plus the depth (and MSAA) attachments
/// that go with it.
///
/// The drawable's colour texture changes every frame; everything else here is
/// reallocated only on resize.
pub struct MetalSurface {
    device: MetalDevice,
    layer: Owned,
    depth: Owned,
    msaa: Option<Owned>,
    width: u32,
    height: u32,
    color_format: NSUInteger,
    sample_count: u32,
}

impl MetalSurface {
    /// Attach a new `CAMetalLayer` to an `NSView` and build the surface.
    /// macOS.
    ///
    /// `ns_view` is what a window toolkit calls the view handle — winit's
    /// `RawWindowHandle::AppKit { ns_view, .. }`, for instance.
    ///
    /// Must be called on the main thread — `setWantsLayer:` and `setLayer:` are
    /// AppKit calls — and fails with [`MetalError::Unsupported`] elsewhere. The
    /// surface it returns can then move to a render thread.
    ///
    /// # Safety
    /// `ns_view` must be null or a live `NSView`.
    pub unsafe fn from_ns_view(
        device: &MetalDevice,
        ns_view: *mut c_void,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        require_main_thread("MetalSurface::from_ns_view")?;
        let cls = metal_layer_class()?;
        // `+layer` is autoreleased; retain it for as long as the surface lives.
        let layer = unsafe { Owned::retain(msg0(cls, sel!("layer"))) }
            .ok_or_else(|| MetalError::Allocation("CAMetalLayer".into()))?;
        let surface = unsafe { Self::build(device, layer, width, height, sample_count) }?;

        let view = ns_view as Id;
        if !view.is_null() {
            let _: () = unsafe { msg1(view, sel!("setWantsLayer:"), YES) };
            let _: () = unsafe { msg1(view, sel!("setLayer:"), surface.layer.id()) };
        }
        Ok(surface)
    }

    /// The surface of a `UIView` whose layer is a `CAMetalLayer`. iOS, iPadOS,
    /// Mac Catalyst, tvOS, and visionOS windowed apps.
    ///
    /// UIKit fixes a view's layer at creation, so the view has to ask for a
    /// Metal one:
    ///
    /// ```swift
    /// final class MetalView: UIView {
    ///     override class var layerClass: AnyClass { CAMetalLayer.self }
    /// }
    /// ```
    ///
    /// Pass `view.bounds.size × contentScaleFactor` as the size. Must be called
    /// on the main thread, where reading `view.layer` is legal, and fails with
    /// [`MetalError::Unsupported`] elsewhere or when the layer is not a
    /// `CAMetalLayer`. The surface it returns can then move to a render thread.
    ///
    /// # Safety
    /// `ui_view` must be a live `UIView`.
    #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
    pub unsafe fn from_ui_view(
        device: &MetalDevice,
        ui_view: *mut c_void,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        require_main_thread("MetalSurface::from_ui_view")?;
        if ui_view.is_null() {
            return Err(MetalError::Unsupported("from_ui_view with a null view".into()));
        }
        let layer: Id = unsafe { msg0(ui_view as Id, sel!("layer")) };
        unsafe { Self::from_layer(device, layer.cast(), width, height, sample_count) }
    }

    /// Adopt a `CAMetalLayer` the app already has — a `UIView` or `NSView`
    /// backed by one, or one the app made and placed itself.
    ///
    /// Sets the layer's device, pixel format (`BGRA8_UNORM_SRGB`),
    /// `framebufferOnly` and drawable size, and leaves its geometry and
    /// `contentsScale` to the view that owns it. The layer is retained for as
    /// long as the surface lives.
    ///
    /// Callable on any thread: a `CAMetalLayer`'s rendering properties are
    /// not UIKit or AppKit state.
    ///
    /// # Safety
    /// `layer` must be null or a live `CALayer`.
    pub unsafe fn from_layer(
        device: &MetalDevice,
        layer: *mut c_void,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Result<Self, MetalError> {
        let _pool = AutoreleasePool::new();
        let cls = metal_layer_class()?;
        let layer = layer as Id;
        let is_metal: Bool = if layer.is_null() {
            0
        } else {
            unsafe { msg1(layer, sel!("isKindOfClass:"), cls) }
        };
        if is_metal == 0 {
            return Err(MetalError::Unsupported(
                "the layer is not a CAMetalLayer — on iOS, override the view's \
                 `layerClass` to return `CAMetalLayer.self`"
                    .into(),
            ));
        }
        let layer = unsafe { Owned::retain(layer) }.ok_or_else(|| MetalError::Allocation("CAMetalLayer".into()))?;
        unsafe { Self::build(device, layer, width, height, sample_count) }
    }

    /// Configure `layer` for this device and allocate the attachments.
    unsafe fn build(
        device: &MetalDevice,
        layer: Owned,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Result<Self, MetalError> {
        // BGRA is the layer's native order; sRGB so the shader can keep writing
        // linear values exactly as it does offscreen.
        let color_format = pixel_format::BGRA8_UNORM_SRGB;
        let _: () = unsafe { msg1(layer.id(), sel!("setDevice:"), device.device_id()) };
        let _: () = unsafe { msg1(layer.id(), sel!("setPixelFormat:"), color_format) };
        let _: () = unsafe { msg1(layer.id(), sel!("setFramebufferOnly:"), YES) };

        let mut surface = Self {
            device: device.clone(),
            layer,
            // Placeholder, replaced by `allocate_attachments` below.
            depth: new_texture_2d(
                device,
                1,
                1,
                pixel_format::DEPTH32_FLOAT,
                texture_usage::RENDER_TARGET,
                storage_mode::PRIVATE,
                1,
                1,
            )?,
            msaa: None,
            width: 0,
            height: 0,
            color_format,
            sample_count: sample_count.max(1),
        };
        if surface.sample_count > 1 && !device.supports_sample_count(surface.sample_count) {
            return Err(MetalError::Unsupported(format!(
                "{}x MSAA — this device does not support that sample count",
                surface.sample_count
            )));
        }
        surface.resize(width, height)?;
        Ok(surface)
    }

    /// Resize the layer's drawables and reallocate the depth/MSAA attachments.
    /// A no-op when the size has not changed.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), MetalError> {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == (self.width, self.height) {
            return Ok(());
        }
        let _pool = AutoreleasePool::new();
        unsafe {
            let _: () = msg1(
                self.layer.id(),
                sel!("setDrawableSize:"),
                CGSize {
                    width: width as f64,
                    height: height as f64,
                },
            );
        }
        self.depth = new_texture_2d(
            &self.device,
            width,
            height,
            pixel_format::DEPTH32_FLOAT,
            texture_usage::RENDER_TARGET,
            storage_mode::PRIVATE,
            self.sample_count,
            1,
        )?;
        self.msaa = if self.sample_count > 1 {
            Some(new_texture_2d(
                &self.device,
                width,
                height,
                self.color_format,
                texture_usage::RENDER_TARGET,
                storage_mode::PRIVATE,
                self.sample_count,
                1,
            )?)
        } else {
            None
        };
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Acquire the next drawable.
    ///
    /// `None` when the layer has no drawable available — every one is in
    /// flight, or the window is off-screen. That is normal; skip the frame.
    pub fn next_frame(&self) -> Option<SurfaceFrame> {
        let _pool = AutoreleasePool::new();
        unsafe {
            let drawable = Owned::retain(msg0(self.layer.id(), sel!("nextDrawable")))?;
            let texture: Id = msg0(drawable.id(), sel!("texture"));
            if texture.is_null() {
                return None;
            }
            let (color, resolve) = match &self.msaa {
                Some(msaa) => (msaa.id(), texture),
                None => (texture, NIL),
            };
            Some(SurfaceFrame {
                attachments: PassAttachments::from_raw(
                    color,
                    resolve,
                    self.depth.id(),
                    drawable.id(),
                    self.width,
                    self.height,
                    self.color_format,
                    pixel_format::DEPTH32_FLOAT,
                    self.sample_count,
                ),
                _drawable: drawable,
            })
        }
    }

    /// Whether the layer is set to `displaySyncEnabled` (vsync). macOS only.
    pub fn vsync(&self) -> bool {
        unsafe {
            let responds: Bool = msg1(
                self.layer.id(),
                sel!("respondsToSelector:"),
                sel!("displaySyncEnabled"),
            );
            if responds == 0 {
                return true;
            }
            let on: Bool = msg0(self.layer.id(), sel!("displaySyncEnabled"));
            on != 0
        }
    }

    /// Turn vsync on or off. Off lets a benchmark run past the refresh rate.
    pub fn set_vsync(&self, enabled: bool) {
        unsafe {
            let responds: Bool = msg1(
                self.layer.id(),
                sel!("respondsToSelector:"),
                sel!("setDisplaySyncEnabled:"),
            );
            if responds != 0 {
                let _: () = msg1(
                    self.layer.id(),
                    sel!("setDisplaySyncEnabled:"),
                    if enabled { YES } else { 0 as Bool },
                );
            }
        }
    }

    /// Drawable size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The colour format the layer presents (`BGRA8_UNORM_SRGB`).
    pub fn color_format(&self) -> NSUInteger {
        self.color_format
    }

    /// The `CAMetalLayer`. Borrowed — do not release.
    pub fn layer_id(&self) -> Id {
        self.layer.id()
    }
}

/// `CAMetalLayer`, or an error naming the framework that did not link.
fn metal_layer_class() -> Result<Id, MetalError> {
    let cls = super::objc::get_class("CAMetalLayer\0");
    if cls.is_null() {
        return Err(MetalError::Unsupported(
            "CAMetalLayer is not available — QuartzCore did not link".into(),
        ));
    }
    Ok(cls)
}

/// AppKit and UIKit throw, or corrupt their state, when a view is touched off
/// the main thread; say so before it happens.
fn require_main_thread(call: &str) -> Result<(), MetalError> {
    if super::objc::is_main_thread() {
        Ok(())
    } else {
        Err(MetalError::Unsupported(format!(
            "{call} touches a view and must run on the main thread; \
             build the surface there and move it to the render thread"
        )))
    }
}

/// One acquired drawable.
///
/// [`MetalRenderer::render`](super::MetalRenderer::render) presents it as part
/// of the same command buffer it draws on, so dropping this after the render
/// call is all the cleanup there is.
pub struct SurfaceFrame {
    attachments: PassAttachments,
    /// Held so the drawable outlives the pool that vended it.
    _drawable: Owned,
}

impl SurfaceFrame {
    /// Attachments for this frame.
    pub fn attachments(&self) -> PassAttachments {
        self.attachments
    }
}

impl std::fmt::Debug for MetalSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSurface")
            .field("size", &(self.width, self.height))
            .field("sample_count", &self.sample_count)
            .finish()
    }
}
