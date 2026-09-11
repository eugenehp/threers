//! Headless offscreen rendering — a batteries-included wrapper that owns the
//! wgpu instance/adapter/device plus a [`Renderer`] and offscreen [`RenderTarget`],
//! so you can render a [`Scene`] to a pixel buffer in a few lines instead of the
//! ~50-line surfaceless-wgpu dance every tool otherwise reimplements.
//!
//! ```no_run
//! use threers::{HeadlessRenderer, Scene, PerspectiveCamera};
//! let mut hr = HeadlessRenderer::builder().size(1920, 1080).supersample(2).build().unwrap();
//! let mut scene = Scene::new();
//! let cam = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 1000.0);
//! let rgba = hr.render_to_rgba(&mut scene, &cam); // tightly-packed RGBA8
//! ```
//!
//! Native-only (uses `pollster` to block on device acquisition + readback).

use std::sync::Arc;

use crate::cameras::Camera;
use crate::renderer::{RenderTarget, Renderer};
use crate::scene::Scene;

/// Prefer a specific GPU vendor when multiple adapters are present.
///
/// On Linux/Windows this selects a Vulkan adapter — NVIDIA (CUDA driver stack)
/// or AMD (ROCm / RADV). macOS ignores this and uses Metal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuVendor {
    #[default]
    Any,
    /// NVIDIA (`0x10DE`) — discrete GeForce/RTX via Vulkan.
    Nvidia,
    /// AMD (`0x1002`) — Radeon via RADV, AMDVLK, or ROCm-backed drivers.
    Amd,
}

impl GpuVendor {
    const NVIDIA_ID: u32 = 0x10DE;
    const AMD_ID: u32 = 0x1002;

    fn vendor_id(self) -> Option<u32> {
        match self {
            Self::Any => None,
            Self::Nvidia => Some(Self::NVIDIA_ID),
            Self::Amd => Some(Self::AMD_ID),
        }
    }

    /// Parse `THREERS_GPU` / `FLY_GPU`: `nvidia`, `cuda`, `amd`, or `rocm`.
    pub fn from_env() -> Self {
        match std::env::var("THREERS_GPU")
            .or_else(|_| std::env::var("FLY_GPU"))
            .ok()
            .as_deref()
        {
            Some("nvidia") | Some("cuda") | Some("NVIDIA") | Some("CUDA") => Self::Nvidia,
            Some("amd") | Some("rocm") | Some("AMD") | Some("ROCM") => Self::Amd,
            _ => Self::Any,
        }
    }
}

fn adapter_score(info: &wgpu::AdapterInfo) -> u32 {
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 100,
        wgpu::DeviceType::IntegratedGpu => 50,
        wgpu::DeviceType::VirtualGpu => 25,
        _ => 0,
    }
}

fn request_headless_adapter(
    instance: &wgpu::Instance,
    config: &HeadlessConfig,
) -> Result<wgpu::Adapter, String> {
    let backends = wgpu::Backends::PRIMARY;
    if let Some(vendor_id) = config.gpu_vendor.vendor_id() {
        let adapters =
            pollster::block_on(instance.enumerate_adapters(backends));
        let mut best: Option<(u32, wgpu::Adapter)> = None;
        for adapter in adapters {
            let info = adapter.get_info();
            if info.vendor != vendor_id {
                continue;
            }
            let score = adapter_score(&info);
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, adapter));
            }
        }
        if let Some((_, adapter)) = best {
            let info = adapter.get_info();
            eprintln!(
                "   gpu       {} ({:?}, {:?}, vendor 0x{:04x})",
                info.name, info.backend, info.device_type, info.vendor
            );
            return Ok(adapter);
        }
        eprintln!(
            "   gpu       warning: no adapter for vendor 0x{vendor_id:04x}, falling back to default"
        );
    }

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: config.power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no suitable wgpu adapter for headless rendering: {e}"))?;
    let info = adapter.get_info();
    eprintln!(
        "   gpu       {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    Ok(adapter)
}

/// Configuration for a [`HeadlessRenderer`]. Prefer [`HeadlessRenderer::builder`].
#[derive(Clone, Debug)]
pub struct HeadlessConfig {
    /// Output width in pixels (before supersampling).
    pub width: u32,
    /// Output height in pixels (before supersampling).
    pub height: u32,
    /// Supersample factor — the scene renders at `size × supersample`. `1` = off.
    pub supersample: u32,
    /// Color target format. `Rgba8UnormSrgb` for images; `Rgba16Float` for HDR.
    pub color_format: wgpu::TextureFormat,
    /// Raise the device's texture-size limits to the adapter maximum — required
    /// for 4K (and 2K × supersample), which exceed wgpu's conservative defaults.
    pub high_resolution: bool,
    /// Enable temporal anti-aliasing (accumulate jittered frames; see
    /// [`Renderer::set_taa`]).
    pub taa: bool,
    /// Adapter selection preference.
    pub power_preference: wgpu::PowerPreference,
    /// When set, pick a discrete adapter from this vendor (Vulkan on Linux).
    pub gpu_vendor: GpuVendor,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            supersample: 1,
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            high_resolution: true,
            taa: false,
            power_preference: wgpu::PowerPreference::HighPerformance,
            gpu_vendor: GpuVendor::Any,
        }
    }
}

/// Fluent builder for [`HeadlessRenderer`].
#[derive(Clone, Debug, Default)]
pub struct HeadlessBuilder {
    config: HeadlessConfig,
}

impl HeadlessBuilder {
    /// Output size in pixels (before supersampling).
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }
    /// Supersample factor (render at `size × factor`, then read back at that
    /// resolution). `1` disables it.
    pub fn supersample(mut self, factor: u32) -> Self {
        self.config.supersample = factor.max(1);
        self
    }
    /// Color target format (default `Rgba8UnormSrgb`).
    pub fn color_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.config.color_format = format;
        self
    }
    /// Raise texture-size limits to the adapter max (default `true`; needed for 4K).
    pub fn high_resolution(mut self, enabled: bool) -> Self {
        self.config.high_resolution = enabled;
        self
    }
    /// Enable temporal anti-aliasing.
    pub fn taa(mut self, enabled: bool) -> Self {
        self.config.taa = enabled;
        self
    }
    /// Adapter power preference.
    pub fn power_preference(mut self, preference: wgpu::PowerPreference) -> Self {
        self.config.power_preference = preference;
        self
    }
    /// Prefer NVIDIA or AMD when multiple GPUs are present (Linux/Windows Vulkan).
    pub fn gpu_vendor(mut self, vendor: GpuVendor) -> Self {
        self.config.gpu_vendor = vendor;
        self
    }
    /// Acquire the GPU and construct the renderer.
    pub fn build(self) -> Result<HeadlessRenderer, String> {
        HeadlessRenderer::new(self.config)
    }
}

/// A headless renderer: owns the GPU device/queue, a [`Renderer`], and an
/// offscreen [`RenderTarget`]. Render with [`render`](Self::render) /
/// [`render_to_rgba`](Self::render_to_rgba); reach the underlying [`Renderer`]
/// via [`renderer`](Self::renderer) for post-fx, TAA, etc.
pub struct HeadlessRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    target: RenderTarget,
    config: HeadlessConfig,
    render_width: u32,
    render_height: u32,
    /// Mapping buffers, recycled. Allocating one per frame costs an allocation
    /// and a free inside the frame loop for no benefit — the size never changes.
    spare: Vec<wgpu::Buffer>,
    /// A readback submitted but not yet drained, for the pipelined path.
    pending: Option<Pending>,
    /// Readbacks in flight on the RGB path, oldest first.
    ///
    /// One deep is not enough. A copy submitted this frame is not finished when
    /// the next frame's draw is queued behind it, so taking it immediately means
    /// waiting on the GPU: measured over 360 frames at 4K, a readback that costs
    /// 9.7 ms on its own costs 34-46 ms in a stream. Holding several lets the
    /// CPU run ahead far enough that the oldest is always already done.
    ///
    /// Measured, 360 frames at 4K with MSAA 4, wall time for the frame stage:
    ///
    /// ```text
    /// depth 1   23.6 s     depth 4   18.5 s
    /// depth 2   22.9 s     depth 6   17.2 s
    ///                      depth 8    8.6 s
    /// ```
    ///
    /// The cost is one mapped buffer per level -- 25 MB a frame at 4K, 100 MB
    /// at 8K -- which is why this is bounded rather than unbounded.
    rgb_queue: std::collections::VecDeque<Pending>,
    /// How many to keep in flight before draining one. READAHEAD overrides it.
    read_ahead: usize,
    /// Averages the supersampled target down to the output size on the GPU.
    /// Built on first use, and only when `supersample > 1`.
    resolve: Option<crate::renderer::Downsampler>,
    /// GPU alpha-strip, built on first use and only when a caller asks for RGB.
    rgb_pack: Option<crate::renderer::rgb_pack::RgbPack>,
    /// Bloom, when a caller has asked for it. Runs just before the pack, on the
    /// same finished frame, so the render path is untouched.
    bloom: Option<crate::renderer::bloom::Bloom>,
    bloom_cfg: Option<(f32, f32, f32)>,
    /// Ambient occlusion, alongside bloom on the same finished frame.
    ssao: Option<crate::renderer::ssao::Ssao>,
    ssao_cfg: Option<(f32, f32, f32, f32)>,
    ssao_lens: (f32, f32),
    /// Shutter fraction for camera motion blur; 0 disables it.
    shutter: f32,
    film: crate::renderer::rgb_pack::FilmGrade,
    /// An RGBA overlay composited over the frame during the pack, so the CPU
    /// never has to write to the frame at all.
    overlay: Option<wgpu::Texture>,
}

/// A texture→buffer copy in flight.
struct Pending {
    buffer: wgpu::Buffer,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    padded: u32,
    unpadded: u32,
    height: u32,
}

impl HeadlessRenderer {
    /// Start a fluent [`HeadlessBuilder`].
    pub fn builder() -> HeadlessBuilder {
        HeadlessBuilder::default()
    }

    /// Construct from a [`HeadlessConfig`] (acquires the GPU; blocks).
    pub fn new(config: HeadlessConfig) -> Result<Self, String> {
        let width = config.width.max(1);
        let height = config.height.max(1);
        let ss = config.supersample.max(1);
        let render_width = width.saturating_mul(ss).max(1);
        let render_height = height.saturating_mul(ss).max(1);

        // `InstanceDescriptor` has no `Default` in wgpu 30 — a display handle is
        // something you either have or deliberately do not, and headless is the
        // second case.
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = request_headless_adapter(&instance, &config)?;

        let adapter_limits = adapter.limits();
        let mut limits = if config.high_resolution {
            wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone())
        } else {
            wgpu::Limits::downlevel_defaults()
        };
        // Downlevel allows 16 sampled textures and samplers per stage; the main
        // pass declares 20 in the fragment stage — material maps, shadow
        // atlases and the environment all bind there. wgpu 0.20 never checked
        // this, so the layout has been over the line the whole time and only
        // wgpu 30's validation says so. `using_resolution` does not cover
        // binding counts, so raise them separately, to whatever the adapter has.
        limits.max_sampled_textures_per_shader_stage =
            adapter_limits.max_sampled_textures_per_shader_stage;
        limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;
        // WHATEVER COMPRESSION THIS ADAPTER HAS, ask for it.
        //
        // A device only exposes features it was asked for, so with
        // `Features::empty()` a BC texture fails wgpu validation at upload time
        // on hardware that supports it perfectly well. Which family is present
        // is not portable -- BC is universal on desktop and missing on most
        // mobile parts, ASTC the other way round, Apple silicon has all three --
        // so the request is the intersection of what is wanted and what the
        // adapter reports, and asking for nothing else keeps the device as
        // widely creatable as it was.
        let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
            // Timestamps for `Renderer::gpu_frame_ms`. Masked by what the
            // adapter reports, like the compression formats above, so asking
            // still cannot fail device creation.
            | wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers headless device"),
                required_features: adapter.features() & compression,
                required_limits: limits,
                ..Default::default()
            },
        ))
        .map_err(|e| format!("wgpu request_device failed: {e:?}"))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            config.color_format,
            render_width,
            render_height,
        );
        renderer.set_taa(config.taa);
        let target = RenderTarget::new(&device, render_width, render_height, config.color_format);

        Ok(Self {
            device,
            queue,
            renderer,
            target,
            config,
            render_width,
            render_height,
            spare: Vec::new(),
            pending: None,
            rgb_queue: std::collections::VecDeque::new(),
            read_ahead: std::env::var("READAHEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8usize)
                .clamp(1, 8),
            rgb_pack: None,
            bloom: None,
            bloom_cfg: None,
            ssao: None,
            ssao_cfg: None,
            ssao_lens: (0.5, 1.777),
            shutter: 0.0,
            film: Default::default(),
            overlay: None,
            resolve: None,
        })
    }

    /// The wrapped [`Renderer`] (for post-fx, TAA, render-target registration…).
    pub fn renderer(&mut self) -> &mut Renderer {
        &mut self.renderer
    }
    /// The GPU device (share it to build geometry/textures/render targets).
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
    /// The GPU queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    /// Enable/disable temporal anti-aliasing (see [`Renderer::set_taa`]).
    pub fn set_taa(&mut self, enabled: bool) {
        self.renderer.set_taa(enabled);
    }
    /// Enable hardware MSAA (`samples <= 1` = off, else 4×) for the opaque forward
    /// pass. Composes with (and is usually preferable to) `supersample`.
    pub fn set_msaa(&mut self, samples: u32) {
        self.renderer.set_msaa(samples);
    }
    /// The render resolution (output size × supersample).
    pub fn render_size(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }
    /// The config this renderer was built with.
    pub fn config(&self) -> &HeadlessConfig {
        &self.config
    }

    /// Render `scene` from `camera` into the offscreen target.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        if self.renderer.msaa() > 1 {
            // Surface-style path so the MSAA opaque pass engages and resolves into
            // the offscreen target's color view.
            let linear = self.config.color_format == wgpu::TextureFormat::Rgba16Float;
            self.renderer
                .render(scene, camera, &self.target.color_view, linear);
        } else {
            self.renderer.render_to(scene, camera, &self.target);
        }
    }

    /// Render, then read the target back as tightly-packed RGBA8 (row-major,
    /// top-left origin) sized [`render_size`](Self::render_size).
    pub fn render_to_rgba(&mut self, scene: &mut Scene, camera: &dyn Camera) -> Vec<u8> {
        self.render(scene, camera);
        self.read_rgba()
    }

    /// Draw the caption showing at `time` seconds over the current target
    /// contents, on the GPU.
    ///
    /// Call it between [`render`](Self::render) and [`read_rgba`](Self::read_rgba),
    /// or use [`render_to_rgba_with_captions`](Self::render_to_rgba_with_captions)
    /// for the whole sequence.
    #[cfg(feature = "captions")]
    pub fn draw_captions(&mut self, overlay: &mut crate::captions::CaptionOverlay, time: f64) {
        let (w, h) = (self.render_width, self.render_height);
        let format = self.target.format;
        // Split the borrow: `renderer` and `target` are separate fields.
        let view = &self.target.color_view;
        self.renderer
            .draw_caption_overlay(overlay, time, w, h, view, format);
    }

    /// Render, blend the caption for `time` seconds on top, and read back RGBA8.
    ///
    /// This is the video-export entry point when you want the captions
    /// composited by the GPU rather than by
    /// [`CaptionPainter`](crate::captions::CaptionPainter) on the CPU. The
    /// result is the same; doing it here keeps the pixels on the GPU for one
    /// fewer pass over the frame.
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Scene, PerspectiveCamera};
    /// # use threers::captions::{CaptionOverlay, CaptionTrack};
    /// # let mut hr = HeadlessRenderer::builder().size(640, 360).build().unwrap();
    /// # let mut scene = Scene::new();
    /// # let camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
    /// let mut overlay = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 2.0, "Hi"))
    ///     .auto_scale(true);
    /// let rgba = hr.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.0);
    /// # let _ = rgba;
    /// ```
    #[cfg(feature = "captions")]
    pub fn render_to_rgba_with_captions(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        overlay: &mut crate::captions::CaptionOverlay,
        time: f64,
    ) -> Vec<u8> {
        self.render(scene, camera);
        self.draw_captions(overlay, time);
        self.read_rgba()
    }

    /// Read the current target contents back as tightly-packed RGBA8, unpadding
    /// the 256-byte row alignment wgpu requires for texture→buffer copies.
    ///
    /// This blocks until the GPU has finished the frame. For a video export,
    /// prefer [`read_rgba_pipelined`](Self::read_rgba_pipelined), which lets the
    /// CPU build the next frame while this one is still being copied.
    pub fn read_rgba(&mut self) -> Vec<u8> {
        let p = self.queue_readback();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.drain(p)
    }

    /// Render, then read back at the configured [`size`](HeadlessBuilder::size),
    /// averaging any [`supersample`](HeadlessBuilder::supersample) on the GPU.
    ///
    /// This is what you usually want when supersampling: `render_to_rgba` hands
    /// back the *oversized* frame and leaves the averaging to you, which means
    /// dragging `factor²` more bytes across the bus and then walking them on the
    /// CPU. At 1920×1080 with `supersample(2)` that walk costs 33 ms against
    /// 4.4 ms for the whole render — so it is done here in the render pass
    /// instead, and only the finished frame is copied back.
    ///
    /// With `supersample(1)` there is nothing to average and this is exactly
    /// [`render_to_rgba`](Self::render_to_rgba).
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Scene, PerspectiveCamera};
    /// let mut hr = HeadlessRenderer::builder().size(1920, 1080).supersample(2).build().unwrap();
    /// # let mut scene = Scene::new();
    /// # let cam = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
    /// let rgba = hr.render_to_rgba_resolved(&mut scene, &cam); // 1920×1080
    /// # let _ = rgba;
    /// ```
    pub fn render_to_rgba_resolved(&mut self, scene: &mut Scene, camera: &dyn Camera) -> Vec<u8> {
        self.render(scene, camera);
        self.read_rgba_resolved()
    }

    /// Read the target back at the configured output size, averaging any
    /// supersample factor on the GPU. See
    /// [`render_to_rgba_resolved`](Self::render_to_rgba_resolved).
    pub fn read_rgba_resolved(&mut self) -> Vec<u8> {
        let Some((w, h)) = self.run_resolve() else {
            // Either there is nothing to average, or the GPU pass declined this
            // format/size. The size this returns is part of the contract, so in
            // the second case fall back rather than hand back an oversized frame.
            let raw = self.read_rgba();
            let factor = self.config.supersample.max(1);
            return if factor > 1 {
                let (rw, rh) = (self.render_width, self.render_height);
                crate::renderer::downsample::average_blocks_srgb(&raw, rw, rh, factor)
            } else {
                raw
            };
        };
        let p = self.queue_readback_from(true, w, h);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.drain(p)
    }

    /// Record the averaging pass, building the downsampler on first use.
    ///
    /// `None` when there is nothing to average — no supersample, or a format the
    /// pass cannot render into — in which case the caller reads the target back
    /// as it stands.
    fn run_resolve(&mut self) -> Option<(u32, u32)> {
        let factor = self.config.supersample.max(1);
        if factor < 2 {
            return None;
        }
        if self.resolve.is_none() {
            self.resolve =
                crate::renderer::Downsampler::new(&self.device, &self.target.color_texture, factor);
        }
        let ds = self.resolve.as_ref()?;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers headless resolve"),
            });
        ds.resolve(&mut enc);
        let size = ds.size();
        self.queue.submit(Some(enc.finish()));
        Some(size)
    }

    /// Queue this frame's readback and return the frame queued on the PREVIOUS
    /// call — `None` the first time. Finish the sequence with
    /// [`finish_readback`](Self::finish_readback).
    ///
    /// `read_rgba` submits a copy and immediately blocks on it, so the GPU is
    /// idle for as long as the CPU takes to build the next frame and the CPU is
    /// idle for as long as the GPU takes to draw. Handing back the previous
    /// frame instead overlaps the two: by the time it is asked for, the copy has
    /// usually already landed and the wait is zero.
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Scene, PerspectiveCamera};
    /// # let mut hr = HeadlessRenderer::builder().size(64, 64).build().unwrap();
    /// # let mut scene = Scene::new();
    /// # let cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    /// for _frame in 0..10 {
    ///     hr.render(&mut scene, &cam);
    ///     if let Some(rgba) = hr.read_rgba_pipelined() { let _ = rgba; }
    /// }
    /// if let Some(rgba) = hr.finish_readback() { let _ = rgba; }
    /// ```
    pub fn read_rgba_pipelined(&mut self) -> Option<Vec<u8>> {
        let next = self.queue_readback();
        let out = self.pending.take().map(|p| {
            // Poll rather than Wait: Wait blocks on everything submitted, which
            // now includes the copy queued one line above — that would serialise
            // exactly what this is here to overlap.
            loop {
                let _ = self.device.poll(wgpu::PollType::Poll);
                match p.rx.try_recv() {
                    Ok(_) => break,
                    Err(std::sync::mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            }
            self.take_pixels(p)
        });
        self.pending = Some(next);
        out
    }

    /// [`HeadlessRenderer::read_rgba_pipelined`] and [`HeadlessRenderer::read_rgba_resolved`] at once: average the
    /// supersample on the GPU, then hand back the PREVIOUS frame at output size.
    ///
    /// The two existing calls do not compose, and a renderer writing a long
    /// animation wants both. `read_rgba_resolved` averages on the GPU but blocks
    /// on its own copy, so the CPU idles through the draw and the GPU idles
    /// through the encode. `read_rgba_pipelined` overlaps them but reads the
    /// target as it stands — the *large* one — leaving the caller to average
    /// `factor²` samples per pixel on the CPU, which is the cost this module
    /// exists to avoid: 33 ms against 4.4 ms for the whole frame at 1080p.
    /// Reaching for it because it was the only pipelined option is an easy
    /// mistake and an expensive one.
    ///
    /// Returns frames sized `size`, not `size × factor`, and shrinks the copy
    /// by `factor²` because it starts from the small texture.
    ///
    /// Finish the sequence with [`finish_readback`](Self::finish_readback), as
    /// with the unresolved version.
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Scene, PerspectiveCamera};
    /// let mut hr = HeadlessRenderer::builder().size(1920, 1080).supersample(2).build().unwrap();
    /// # let mut scene = Scene::new();
    /// # let cam = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
    /// for _frame in 0..10 {
    ///     hr.render(&mut scene, &cam);
    ///     // 1920×1080, averaged on the GPU, one frame behind.
    ///     if let Some(rgba) = hr.read_rgba_resolved_pipelined() { let _ = rgba; }
    /// }
    /// if let Some(rgba) = hr.finish_readback() { let _ = rgba; }
    /// ```
    pub fn read_rgba_resolved_pipelined(&mut self) -> Option<Vec<u8>> {
        let factor = self.config.supersample.max(1);
        let resolved = self.run_resolve();
        // The downsampler is built once and cached, so whether the GPU pass is
        // available does not change between frames; the frame handed back below
        // was queued the same way as the one queued here.
        let on_gpu = resolved.is_some();
        let next = match resolved {
            Some((w, h)) => self.queue_readback_from(true, w, h),
            None => self.queue_readback(),
        };
        let out = self.pending.take().map(|p| {
            // Poll rather than Wait, for the reason given on read_rgba_pipelined.
            loop {
                let _ = self.device.poll(wgpu::PollType::Poll);
                match p.rx.try_recv() {
                    Ok(_) => break,
                    Err(std::sync::mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            }
            let raw = self.take_pixels(p);
            // Only when the GPU declined the format or size. The output size is
            // part of the contract either way.
            if !on_gpu && factor > 1 {
                let (rw, rh) = (self.render_width, self.render_height);
                crate::renderer::downsample::average_blocks_srgb(&raw, rw, rh, factor)
            } else {
                raw
            }
        });
        self.pending = Some(next);
        out
    }

    /// Drain the last frame left in flight by
    /// [`read_rgba_pipelined`](Self::read_rgba_pipelined).
    pub fn finish_readback(&mut self) -> Option<Vec<u8>> {
        let p = self.pending.take()?;
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Some(self.drain(p))
    }

    fn drain(&mut self, p: Pending) -> Vec<u8> {
        let _ = p.rx.recv();
        self.take_pixels(p)
    }

    /// Read a cube render target back as six linear-light RGBA faces.
    ///
    /// A cube rendered from a point in the scene is a reflection probe, and a
    /// probe is not usable as image-based lighting until it has been filtered
    /// per roughness: sampled raw, a surface at roughness 0.5 reflects a
    /// mirror-sharp environment, which looks worse than the crude static one it
    /// replaced. wgpu exposes a render target's cube view with a single mip, so
    /// there is nothing for a roughness-to-mip mapping to read.
    ///
    /// Reading it back is what makes the existing prefilter usable on it: the
    /// caller hands these faces to `PmremGenerator` and gets the same
    /// roughness-aware environment a static cube gets. A probe is small — six
    /// 48-pixel faces is 55 kB — so this costs far less than it sounds.
    ///
    /// Faces come back in cube order (+X, -X, +Y, -Y, +Z, -Z), decoded to
    /// LINEAR light: the target is usually sRGB, and lighting maths in an
    /// sRGB-encoded value is wrong in a way that looks merely "a bit dark".
    pub fn read_cube_faces(&self, rt: &crate::renderer::CubeRenderTarget) -> Option<[Vec<f32>; 6]> {
        let n = rt.side as usize;
        let bpp = 4usize;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let unpadded = n * bpp;
        let padded = unpadded.div_ceil(align) * align;
        let size = (padded * n) as u64;

        let srgb = matches!(
            rt.format,
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let bgra = matches!(
            rt.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers cube readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        for (face, slot) in out.iter_mut().enumerate() {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("threers cube readback"),
                });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &rt.color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded as u32),
                        rows_per_image: Some(n as u32),
                    },
                },
                wgpu::Extent3d {
                    width: rt.side,
                    height: rt.side,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(Some(enc.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().ok()?.is_err() {
                return None;
            }
            {
                let view = buf.slice(..).get_mapped_range().expect("buffer range is mapped");
                let mut f = Vec::with_capacity(n * n * 4);
                for y in 0..n {
                    let row = &view[y * padded..y * padded + unpadded];
                    for x in 0..n {
                        let p = &row[x * 4..x * 4 + 4];
                        let (r, g, b) = if bgra {
                            (p[2], p[1], p[0])
                        } else {
                            (p[0], p[1], p[2])
                        };
                        for c in [r, g, b] {
                            let v = c as f32 / 255.0;
                            f.push(if srgb { srgb_to_linear(v) } else { v });
                        }
                        f.push(1.0);
                    }
                }
                *slot = f;
            }
            buf.unmap();
        }
        Some(out)
    }

    /// Turn bloom on for the RGB readback path: `strength` scales what is added
    /// back, `threshold` is the linear luminance a pixel must exceed to bloom
    /// at all, and `radius` widens the blur.
    ///
    /// Only the RGB path. That is where the frame is already being read by a
    /// compute shader on its way to the encoder, so the composite costs a
    /// sample rather than another pass over the picture.
    pub fn set_bloom(&mut self, strength: f32, threshold: f32, radius: f32) {
        let cfg = (strength, threshold, radius);
        if self.bloom_cfg != Some(cfg) {
            self.bloom_cfg = Some(cfg);
            self.bloom = None;
            self.rgb_pack = None;
        }
    }

    /// Turn ambient occlusion on for the RGB readback path. `radius` is in
    /// world units and `near`/`far` MUST match the camera the frame is rendered
    /// with — the depth linearisation is wrong otherwise, and wrong silently.
    /// Cheap to call every frame: only a change of strength or radius rebuilds
    /// anything, and a new depth range is a buffer write.
    pub fn set_ssao(
        &mut self,
        strength: f32,
        radius: f32,
        near: f32,
        far: f32,
        tan_half_fov: f32,
        aspect: f32,
    ) {
        let rebuild = self
            .ssao_cfg
            .map(|(s, r, _, _)| s != strength || r != radius)
            .unwrap_or(true);
        self.ssao_cfg = Some((strength, radius, near, far));
        self.ssao_lens = (tan_half_fov, aspect);
        if rebuild {
            self.ssao = None;
            self.rgb_pack = None;
        } else if let Some(a) = self.ssao.as_ref() {
            a.set_range(
                &self.queue,
                near,
                far,
                radius,
                strength,
                tan_half_fov,
                aspect,
            );
        }
    }

    /// Set an RGBA overlay composited over the top-left of every frame on the
    /// RGB readback path — a masthead, a lower third.
    ///
    /// Doing this here rather than after readback is what allows the frame to
    /// be handed to an encoder without copying it: a CPU composite needs a
    /// writable frame, and the frame the GPU wrote is mapped read-only.
    pub fn set_overlay(&mut self, width: u32, height: u32, rgba: &[u8]) {
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            self.overlay = None;
            self.rgb_pack = None;
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers headless overlay"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.overlay = Some(tex);
        self.rgb_pack = None;
    }

    /// Camera motion blur on the RGB readback path. `shutter` is the fraction
    /// of the frame interval the shutter is open — 0.5 is the 180-degree
    /// convention, 0 is off.
    pub fn set_motion_blur(&mut self, shutter: f32) {
        if self.shutter != shutter {
            self.shutter = shutter;
            self.rgb_pack = None;
        }
    }

    /// Vignette and grain on the RGB readback path -- the two things that read
    /// as "photograph" rather than "render".
    ///
    /// They are applied in the pack pass, which already touches every pixel, and
    /// BEFORE the overlay, so a masthead composited on top stays clean. Both at
    /// zero is bit-for-bit the frame without them.
    ///
    /// The seed advances itself per packed frame, so grain moves.
    pub fn set_film_grade(&mut self, vignette: f32, grain: f32) {
        if self.film.vignette != vignette || self.film.grain != grain {
            self.film.vignette = vignette;
            self.film.grain = grain;
            self.rgb_pack = None;
        }
    }

    /// Whether [`read_rgb_resolved_pipelined`](Self::read_rgb_resolved_pipelined)
    /// can serve this target. Ask once, before the first frame: the two
    /// readbacks share a pipeline slot and cannot be alternated.
    pub fn rgb_readback_supported(&self) -> bool {
        matches!(
            self.target.color_texture.format(),
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
        )
    }
}

/// A frame still sitting in the buffer the GPU wrote it to.
///
/// The ordinary readback copies the mapped range into a `Vec` and hands that
/// on. At 8K that is 99.5 MB memcpy'd per frame for no reason: the bytes are
/// already in memory the CPU can read, and the only thing done with them is a
/// write to a pipe, which reads them once more. Handing the buffer over instead
/// removes a full-frame copy from every frame.
///
/// The buffer belongs to the renderer's pool and MUST come back — call
/// [`HeadlessRenderer::recycle`] when the bytes have been consumed, or the pool
/// allocates a fresh 99.5 MB every frame instead of reusing four.
pub struct MappedFrame {
    buffer: wgpu::Buffer,
    len: usize,
}

impl MappedFrame {
    /// The frame's bytes. Valid until this is recycled.
    pub fn bytes(&self) -> wgpu::BufferView {
        self.buffer.slice(..self.len as u64).get_mapped_range().expect("buffer range is mapped")
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Give the buffer back. Unmaps first, which is what makes it reusable.
    pub fn into_buffer(self) -> wgpu::Buffer {
        self.buffer.unmap();
        self.buffer
    }
}

impl HeadlessRenderer {
    /// Like [`read_rgb_resolved_pipelined`](Self::read_rgb_resolved_pipelined),
    /// but without copying the frame out of the buffer the GPU wrote it to.
    ///
    /// The returned [`MappedFrame`] must be handed back with
    /// [`recycle`](Self::recycle) once its bytes have been used.
    pub fn read_rgb_frame(&mut self) -> Option<MappedFrame> {
        self.queue_rgb_readback()?;
        if self.rgb_queue.len() <= self.read_ahead {
            return None;
        }
        self.drain_rgb_frame()
    }

    /// Take the oldest readback in flight without queuing another.
    ///
    /// The tail of a sequence: `read_ahead` frames are still in the queue when
    /// the last one has been drawn, and every one of them is a frame the caller
    /// asked for. Call until it returns `None`.
    pub fn drain_rgb_frame(&mut self) -> Option<MappedFrame> {
        let p = self.rgb_queue.pop_front()?;
        self.await_pending(&p);
        Some(MappedFrame {
            buffer: p.buffer,
            len: p.unpadded as usize,
        })
    }

    /// As [`drain_rgb_frame`](Self::drain_rgb_frame), copying the bytes out.
    pub fn drain_rgb_pixels(&mut self) -> Option<Vec<u8>> {
        let p = self.rgb_queue.pop_front()?;
        self.await_pending(&p);
        Some(self.take_pixels(p))
    }

    /// Return a buffer taken by [`read_rgb_frame`](Self::read_rgb_frame).
    pub fn recycle(&mut self, frame: MappedFrame) {
        self.spare.push(frame.into_buffer());
    }
}

impl HeadlessRenderer {
    /// Like [`read_rgba_resolved_pipelined`](Self::read_rgba_resolved_pipelined),
    /// but the frame comes back as tightly-packed RGB — three bytes a pixel,
    /// no alpha, no row padding.
    ///
    /// The strip happens in a compute pass on the frame the GPU already holds,
    /// so a quarter less data crosses the readback and the CPU never touches
    /// the pixels twice. Doing the same pack on the CPU costs far more than the
    /// encoder saves; the numbers are in [`rgb_pack`](crate::renderer::rgb_pack).
    ///
    /// Returns `None` if the colour format cannot be packed, or if
    /// supersampling is on but the GPU downsampler is unavailable — in either
    /// case the caller should stay on the RGBA path. Decide once, before the
    /// sequence starts: switching mid-stream would drop the frame in flight.
    /// Queue this frame's post passes and readback, leaving the PREVIOUS
    /// frame in `self.pending` for a caller to take. Shared by the copying and
    /// zero-copy readbacks so the two cannot drift apart.
    fn queue_rgb_readback(&mut self) -> Option<usize> {
        let factor = self.config.supersample.max(1);
        let resolved = self.run_resolve();
        if factor > 1 && resolved.is_none() {
            return None;
        }
        let (w, h) = resolved.unwrap_or((self.render_width, self.render_height));

        // Rebuild when the target resizes; the bind groups hold views of it.
        if self.rgb_pack.as_ref().map(|p| p.size()) != Some((w, h))
            || self.bloom.as_ref().map(|b| b.src_size()) != self.bloom_cfg.map(|_| (w, h))
            || (self.ssao_cfg.is_some() && self.ssao.is_none())
        {
            let src: &wgpu::Texture = match (resolved.is_some(), self.resolve.as_ref()) {
                (true, Some(ds)) => ds.texture(),
                _ => &self.target.color_texture,
            };
            // Raw pointer dance avoided: build bloom first, then the pack that
            // borrows its view, and only then store both.
            let bloom = self.bloom_cfg.and_then(|(_, threshold, radius)| {
                crate::renderer::bloom::Bloom::new(&self.device, src, threshold, radius)
            });
            let strength = self.bloom_cfg.map(|c| c.0).unwrap_or(0.0);
            // Whichever depth buffer this frame was actually drawn into.
            // `render` branches: with MSAA it takes a surface-style path and
            // writes the renderer's own depth, without it `render_to` writes
            // the target's. Occlusion built on the other one reads an empty
            // buffer and produces a uniformly white factor — which looks
            // exactly like "SSAO is on and subtle", and is why turning it on
            // changed nothing at all on the default, non-MSAA path.
            let depth_src: &wgpu::Texture = if self.renderer.msaa() > 1 {
                self.renderer.depth_texture()
            } else {
                &self.target.depth_texture
            };
            let ssao = self.ssao_cfg.and_then(|(strength, radius, near, far)| {
                let depth = depth_src;
                crate::renderer::ssao::Ssao::new(
                    &self.device,
                    depth,
                    near,
                    far,
                    radius,
                    strength,
                    self.ssao_lens.0,
                    self.ssao_lens.1,
                )
            });
            let (inv_vp, prev_vp) = self.renderer.reprojection();
            self.rgb_pack = crate::renderer::rgb_pack::RgbPack::with_post_film(
                &self.device,
                src,
                bloom.as_ref().map(|b| b.view()),
                strength,
                ssao.as_ref().map(|s| s.view()),
                Some(depth_src),
                inv_vp,
                prev_vp,
                self.shutter,
                self.overlay.as_ref(),
                self.film,
            );
            self.bloom = bloom;
            self.ssao = ssao;
        }
        let pack = self.rgb_pack.as_ref()?;
        // Tick the grain so it moves frame to frame. Free when grain is off:
        // the shader branches out on it either way, and this is eight bytes.
        if self.film.grain > 0.0 {
            self.film.seed += 1.0;
            pack.set_grain_seed(&self.queue, self.film.seed);
        }
        let len = pack.len();
        // buffer_to_buffer copies must be a whole number of words; the few
        // bytes of slack past `len` are the 4-pixel group padding, trimmed by
        // `take_pixels` because `unpadded` states the honest length.
        let copy = len.div_ceil(4) * 4;

        let buffer = match self.spare.iter().position(|b| b.size() == copy as u64) {
            Some(i) => self.spare.swap_remove(i),
            None => {
                self.spare.clear();
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers headless rgb readback"),
                    size: copy as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers headless rgb readback"),
            });
        // This frame's reprojection, before anything samples it.
        if self.shutter > 0.0 {
            let (inv_vp, prev_vp) = self.renderer.reprojection();
            pack.set_reprojection(&self.queue, inv_vp, prev_vp);
        }
        // Bloom first: the pack reads its output.
        if let Some(b) = self.bloom.as_ref() {
            b.record(&mut enc);
        }
        if let Some(a) = self.ssao.as_ref() {
            a.record(&mut enc);
        }
        pack.pack(&mut enc);
        enc.copy_buffer_to_buffer(pack.buffer(), 0, &buffer, 0, copy as u64);
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // One linear "row": `take_pixels` hands back exactly `unpadded` bytes.
        let next = Pending {
            buffer,
            rx,
            padded: copy as u32,
            unpadded: len as u32,
            height: 1,
        };

        // Queued, not taken: the caller drains the OLDEST once enough are in
        // flight. `pending` still gets it so `finish_readback` keeps working
        // for callers that never touch the queue.
        self.rgb_queue.push_back(next);
        Some(len)
    }

    /// Copying readback: the frame comes back as a `Vec`.
    pub fn read_rgb_resolved_pipelined(&mut self) -> Option<Vec<u8>> {
        self.queue_rgb_readback()?;
        if self.rgb_queue.len() <= self.read_ahead {
            return None;
        }
        self.drain_rgb_pixels()
    }

    fn await_pending(&self, p: &Pending) {
        loop {
            let _ = self.device.poll(wgpu::PollType::Poll);
            match p.rx.try_recv() {
                Ok(_) => break,
                Err(std::sync::mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }

    fn take_pixels(&mut self, p: Pending) -> Vec<u8> {
        let data = p.buffer.slice(..).get_mapped_range().expect("buffer range is mapped");
        let (padded, unpadded, h) = (p.padded as usize, p.unpadded as usize, p.height as usize);
        let pixels = if padded == unpadded {
            // The common case at sane widths — 1920 * 4 is already 256-aligned —
            // and one memcpy instead of `height` of them.
            data.to_vec()
        } else {
            let mut v = Vec::with_capacity(unpadded * h);
            for row in data.chunks(padded) {
                v.extend_from_slice(&row[..unpadded]);
            }
            v
        };
        drop(data);
        p.buffer.unmap();
        self.spare.push(p.buffer);
        pixels
    }

    fn queue_readback(&mut self) -> Pending {
        let (w, h) = (self.render_width, self.render_height);
        self.queue_readback_from(false, w, h)
    }

    /// Copy the render target — or, when `resolved`, the downsampler's output —
    /// into a mapped buffer.
    ///
    /// Takes a flag rather than a `&wgpu::Texture` so the source can be borrowed
    /// from `self` at the point of use; the size is passed because the caller
    /// already knows it and the two must agree.
    fn queue_readback_from(&mut self, resolved: bool, w: u32, h: u32) -> Pending {
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let size = (padded * h) as u64;

        let buffer = match self.spare.iter().position(|b| b.size() == size) {
            Some(i) => self.spare.swap_remove(i),
            None => {
                self.spare.clear(); // the target resized; the old ones are useless
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers headless readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers headless readback"),
            });
        let source = match (resolved, self.resolve.as_ref()) {
            (true, Some(ds)) => ds.texture(),
            _ => &self.target.color_texture,
        };
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        Pending {
            buffer,
            rx,
            padded,
            unpadded,
            height: h,
        }
    }
}

/// sRGB byte value to linear light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
