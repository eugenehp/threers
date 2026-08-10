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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: config.power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no suitable wgpu adapter for headless rendering".to_string())?;

        let limits = if config.high_resolution {
            wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits())
        } else {
            wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers headless device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
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
            self.renderer.render(scene, camera, &self.target.color_view, linear);
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

    /// Read the current target contents back as tightly-packed RGBA8, unpadding
    /// the 256-byte row alignment wgpu requires for texture→buffer copies.
    pub fn read_rgba(&self) -> Vec<u8> {
        let (w, h) = (self.render_width, self.render_height);
        let unpadded = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers headless readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers headless readback"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.target.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
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
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = buffer.slice(..).get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * h) as usize);
        for row in data.chunks(padded as usize) {
            pixels.extend_from_slice(&row[..unpadded as usize]);
        }
        drop(data);
        buffer.unmap();
        pixels
    }
}
