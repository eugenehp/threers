//! Getting a device, and the limits you have to ask for.
//!
//! Every example in this crate used to open with the same thirty lines:
//! an instance, an adapter, a `Limits` built up from the downlevel defaults
//! with two fields raised, a device, and two `Arc`s. The two raised fields are
//! not optional and not guessable — `Renderer::new` fails wgpu's pipeline-layout
//! validation without them — so getting this wrong produces an error from
//! inside wgpu about a layout nobody wrote.
//!
//! ```no_run
//! # async fn demo() -> Result<(), String> {
//! use threers::gpu::GpuContext;
//!
//! let gpu = GpuContext::headless().await?;          // no window, no surface
//! let renderer = threers::Renderer::new(
//!     gpu.device.clone(), gpu.queue.clone(), wgpu::TextureFormat::Rgba8Unorm, 800, 600);
//! # let _ = renderer;
//! # Ok(()) }
//! ```
//!
//! On the web, `crate::wasm::open_browser_gpu` is the counterpart (it only
//! exists on `wasm32`, so this is not a link): it has to
//! deal with WebGPU-or-WebGL2 and with a canvas that can only ever hand out one
//! kind of context, neither of which applies natively.

use std::sync::Arc;

/// An adapter and the device opened on it, with the instance that made them.
///
/// The instance is kept because dropping it invalidates nothing on most
/// backends and everything on WebGL2, where the context belongs to it.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// Features taken whenever the adapter has them.
///
/// `RG11B10UFLOAT_RENDERABLE` makes `rg11b10ufloat` a render target, which is
/// what lets the fluid's volume carry its dye in half the memory — see
/// [`crate::fluid::FluidFormats::volume_dye`]. It is in Metal, Vulkan and
/// D3D12, and in WebGL2 with `EXT_color_buffer_float`.
///
/// `TIMESTAMP_QUERY` is what [`GpuTimer`](crate::renderer::gpu_timer::GpuTimer)
/// needs. Asking for it up front matters: a device opened without it cannot
/// be told to measure later, and on a machine with anything else running it
/// is the only way to get a number that means something.
const OPTIONAL_FEATURES: wgpu::Features =
    wgpu::Features::RG11B10UFLOAT_RENDERABLE.union(wgpu::Features::TIMESTAMP_QUERY);

impl GpuContext {
    /// A device with no surface — for headless rendering, compute, and tests.
    pub async fn headless() -> Result<Self, String> {
        let instance = instance(backends());
        let adapter = request_adapter(&instance, None).await?;
        Self::open(instance, adapter).await
    }

    /// A device and a surface for something that can be drawn into — a window,
    /// or a canvas.
    ///
    /// The surface is created before the adapter is requested and passed as
    /// `compatible_surface`, which is what makes the adapter one that can
    /// actually present to it.
    pub async fn with_surface(
        target: wgpu::SurfaceTarget<'static>,
    ) -> Result<(Self, wgpu::Surface<'static>), String> {
        let instance = instance(backends());
        let surface = instance
            .create_surface(target)
            .map_err(|e| format!("no surface: {e}"))?;
        let adapter = request_adapter(&instance, Some(&surface)).await?;
        let gpu = Self::open(instance, adapter).await?;
        Ok((gpu, surface))
    }

    /// Open a device on an adapter you already have.
    pub async fn open(instance: wgpu::Instance, adapter: wgpu::Adapter) -> Result<Self, String> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("threers device"),
                // Nothing here is required — each one is a format or a query
                // that something can use if it is there and does without if
                // it is not, so asking for the intersection costs nothing and
                // refuses no adapter.
                required_features: OPTIONAL_FEATURES & adapter.features(),
                required_limits: renderer_limits(&adapter),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        #[cfg(target_arch = "wasm32")]
        report_uncaptured_errors(&device, "threers");
        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Open on a `CAMetalLayer`, which is how iOS and iPadOS hand you a
    /// drawable surface.
    ///
    /// UIKit has no window handle to pass: a view that draws with Metal has a
    /// `CAMetalLayer`, and that layer *is* the surface. Take the pointer from
    /// the view's `layer` property — after setting the view's `layerClass` to
    /// `CAMetalLayer` — and hand it over here.
    ///
    /// ```swift
    /// // In the UIView subclass
    /// override class var layerClass: AnyClass { CAMetalLayer.self }
    /// // …then, once, from Swift into the Rust staticlib:
    /// fluid_create(Unmanaged.passUnretained(self.layer).toOpaque(), …)
    /// ```
    ///
    /// # Safety
    ///
    /// `layer` must be a live `CAMetalLayer` that outlives the returned
    /// surface. Nothing here retains it: the view owns its layer and will
    /// free it when it goes, and a surface pointing at a freed layer is the
    /// one mistake this cannot check for.
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub async unsafe fn from_metal_layer(
        layer: *mut std::ffi::c_void,
    ) -> Result<(Self, wgpu::Surface<'static>), String> {
        let instance = instance(backends());
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer))
                .map_err(|e| format!("no surface from the layer: {e}"))?
        };
        let adapter = request_adapter(&instance, Some(&surface)).await?;
        let gpu = Self::open(instance, adapter).await?;
        Ok((gpu, surface))
    }

    /// [`GpuContext::headless`], run to completion. **Native only** — a
    /// browser's main thread cannot be blocked.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless_blocking() -> Result<Self, String> {
        pollster::block_on(Self::headless())
    }

    /// [`GpuContext::with_surface`], run to completion. **Native only.**
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_surface_blocking(
        target: wgpu::SurfaceTarget<'static>,
    ) -> Result<(Self, wgpu::Surface<'static>), String> {
        pollster::block_on(Self::with_surface(target))
    }

    /// A surface configuration for `surface` at this size, with a format from
    /// the surface's own capabilities.
    ///
    /// `prefer_srgb` picks between the two conventions, and which you want
    /// depends on who is doing the transfer function. The renderer encodes on
    /// the way out and wants an sRGB surface; a pass that already hands back
    /// display-referred values wants a plain one, or the encode happens twice.
    pub fn surface_config(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
        prefer_srgb: bool,
    ) -> wgpu::SurfaceConfiguration {
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb() == prefer_srgb)
            .unwrap_or(caps.formats[0]);
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        }
    }

    /// What the adapter calls itself, for a log line.
    pub fn describe(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }
}

/// The limits [`crate::Renderer`] needs, raised from the downlevel defaults.
///
/// Two fields and no more, each raised to whatever the adapter actually has.
/// The main pass declares twenty sampled textures in the fragment stage —
/// material maps, the shadow atlases, the environment — against a downlevel
/// allowance of sixteen, and `Renderer::new` fails wgpu 30's validation rather
/// than degrading. `using_resolution` is for the other one that bites: the
/// shadow atlas is 4096 and the downlevel cap on a 2D texture is 2048.
///
/// This is [`crate::HeadlessRenderer`]'s rule, factored out — with `.max()`
/// rather than a plain assignment, so an adapter that somehow reports *less*
/// than the downlevel floor cannot drag the request below it.
///
/// The floor is WebGL2's on a GL adapter. The general downlevel floor asks for
/// compute (65535 workgroups per dimension, among others) and WebGL2 has none,
/// so starting from it makes `request_device` refuse every WebGL2 adapter —
/// which is what [`GpuContext::open`] on one from `open_browser_gpu`'s
/// fallback used to do.
/// Print a device's uncaptured errors to the browser console.
///
/// WebGPU validates pipelines, bind groups and passes *asynchronously*: a
/// rejected one throws nothing and fails no call, it just never draws. Unless
/// something listens, the result is a black canvas at a full frame rate — and
/// inside a worker the browser does not print these either. Every web device
/// this crate opens installs this, and an embedder that opens its own device
/// should too.
///
/// At most 16 messages, then one line saying the rest are dropped: a broken
/// pipeline fails on every frame, and a console of identical lines hides the
/// first one, which is usually the cause.
///
/// wasm only. Natively wgpu's default is to panic on an uncaptured error,
/// which is louder than a log line and is kept.
#[cfg(target_arch = "wasm32")]
pub fn report_uncaptured_errors(device: &wgpu::Device, label: &'static str) {
    use std::sync::atomic::{AtomicU32, Ordering};
    const MAX: u32 = 16;
    let seen = AtomicU32::new(0);
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        let n = seen.fetch_add(1, Ordering::Relaxed);
        let msg = if n < MAX {
            format!("[{label}] GPU error: {e}")
        } else if n == MAX {
            format!("[{label}] further GPU errors are not printed")
        } else {
            return;
        };
        web_sys::console::error_1(&msg.into());
    }));
}

pub fn renderer_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let adapter_limits = adapter.limits();
    let floor = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::downlevel_defaults()
    };
    let mut limits = floor.using_resolution(adapter_limits.clone());
    limits.max_sampled_textures_per_shader_stage = adapter_limits
        .max_sampled_textures_per_shader_stage
        .max(limits.max_sampled_textures_per_shader_stage);
    limits.max_samplers_per_shader_stage = adapter_limits
        .max_samplers_per_shader_stage
        .max(limits.max_samplers_per_shader_stage);
    limits
}

fn backends() -> wgpu::Backends {
    if cfg!(target_arch = "wasm32") {
        wgpu::Backends::BROWSER_WEBGPU
    } else {
        wgpu::Backends::PRIMARY
    }
}

fn instance(backends: wgpu::Backends) -> wgpu::Instance {
    // wgpu 30 dropped `Default` here; the display handle is only consulted by
    // GLES/Wayland, not Vulkan, Metal or DX12.
    let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
    d.backends = backends;
    wgpu::Instance::new(d)
}

async fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter, String> {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| format!("no adapter: {e}"))
}

/// Getting a device in a test, and skipping cleanly when there is not one.
///
/// Every GPU test in this crate opens by asking for an adapter and bailing out
/// if there is none — CI machines often have no GPU, and a test suite that
/// fails there is one nobody runs. The bail-out has to be *visible*, though: a
/// test that silently returns is indistinguishable from one that passed, which
/// is how a whole suite can quietly stop testing anything.
pub mod testing {
    // Native only, as the one function here is: a browser's main thread cannot
    // block on a device request, so there is nothing to import on wasm either.
    #[cfg(not(target_arch = "wasm32"))]
    use super::GpuContext;

    /// A context, or `None` after printing why. **Native only.**
    ///
    /// ```no_run
    /// # fn a_test() {
    /// let Some(gpu) = threers::gpu::testing::context("my test") else { return };
    /// # let _ = gpu;
    /// # }
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn context(what: &str) -> Option<GpuContext> {
        match GpuContext::headless_blocking() {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("skipping {what}: {e}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two raised limits are the whole reason this module exists, so they
    /// are worth an assertion rather than a comment: a refactor that drops
    /// them produces a validation failure inside wgpu, a long way from here.
    #[test]
    fn the_renderer_limits_raise_the_two_that_matter() {
        let Some(gpu) = testing::context("limits") else {
            return;
        };
        let limits = renderer_limits(&gpu.adapter);
        let downlevel = wgpu::Limits::downlevel_defaults();
        let adapter = gpu.adapter.limits();
        // Never below the floor, and never below what the adapter offers —
        // the second is the part that matters, since the floor is what the
        // main pass's binding count exceeds.
        for (name, got, floor, have) in [
            (
                "sampled textures",
                limits.max_sampled_textures_per_shader_stage,
                downlevel.max_sampled_textures_per_shader_stage,
                adapter.max_sampled_textures_per_shader_stage,
            ),
            (
                "samplers",
                limits.max_samplers_per_shader_stage,
                downlevel.max_samplers_per_shader_stage,
                adapter.max_samplers_per_shader_stage,
            ),
        ] {
            assert!(got >= floor, "{name}: {got} is below the downlevel {floor}");
            assert!(
                got >= have,
                "{name}: {got} leaves the adapter's {have} unused"
            );
        }
        assert!(
            limits.max_texture_dimension_2d >= 4096,
            "the shadow atlas is 4096 and this allows {}",
            limits.max_texture_dimension_2d
        );
    }

    /// A context that cannot build a renderer is no use, so build one.
    #[test]
    fn a_headless_context_can_carry_a_renderer() {
        let Some(gpu) = testing::context("headless context") else {
            return;
        };
        assert!(!gpu.describe().is_empty());
        let _renderer = crate::Renderer::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
            64,
            64,
        );
    }
}
