//! Batteries-included offscreen rendering with Metal.
//!
//! The Metal counterpart of [`crate::renderer::HeadlessRenderer`]: owns the
//! device, the renderer and an offscreen target, and hands back RGBA bytes.

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::renderer::{MetalRenderStats, MetalRenderer};
use super::target::MetalRenderTarget;
use crate::cameras::Camera;
use crate::scene::Scene;

/// Configuration for a [`MetalHeadlessRenderer`].
#[derive(Clone, Copy, Debug)]
pub struct MetalHeadlessConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// MSAA sample count; 1 disables it.
    pub sample_count: u32,
    /// Colour attachment format. `RGBA8_UNORM_SRGB` by default, matching the
    /// wgpu headless renderer.
    pub color_format: NSUInteger,
}

impl Default for MetalHeadlessConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            sample_count: 1,
            color_format: pixel_format::RGBA8_UNORM_SRGB,
        }
    }
}

/// Fluent builder for [`MetalHeadlessRenderer`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalHeadlessBuilder {
    config: MetalHeadlessConfig,
}

impl MetalHeadlessBuilder {
    /// Output size in pixels.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// MSAA sample count (1, 2, 4 or 8; 1 disables).
    pub fn msaa(mut self, sample_count: u32) -> Self {
        self.config.sample_count = sample_count.max(1);
        self
    }

    /// Colour attachment format.
    pub fn color_format(mut self, format: NSUInteger) -> Self {
        self.config.color_format = format;
        self
    }

    /// Acquire the GPU, compile the shaders and allocate the target.
    pub fn build(self) -> Result<MetalHeadlessRenderer, MetalError> {
        MetalHeadlessRenderer::new(self.config)
    }
}

/// Renders a [`Scene`] to a pixel buffer with Metal, no window involved.
///
/// ```no_run
/// # #[cfg(all(feature = "metal", target_os = "macos"))] {
/// use threers::metal::MetalHeadlessRenderer;
/// use threers::{PerspectiveCamera, Scene};
///
/// let mut hr = MetalHeadlessRenderer::builder().size(512, 512).msaa(4).build().unwrap();
/// let mut scene = Scene::new();
/// let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
/// let rgba = hr.render_to_rgba(&mut scene, &camera).unwrap();
/// # }
/// ```
pub struct MetalHeadlessRenderer {
    device: MetalDevice,
    renderer: MetalRenderer,
    target: MetalRenderTarget,
    config: MetalHeadlessConfig,
    last_stats: MetalRenderStats,
}

impl MetalHeadlessRenderer {
    /// Start a fluent builder.
    pub fn builder() -> MetalHeadlessBuilder {
        MetalHeadlessBuilder::default()
    }

    /// Construct from a config, acquiring the GPU.
    pub fn new(config: MetalHeadlessConfig) -> Result<Self, MetalError> {
        let device = MetalDevice::new()?;
        let renderer = MetalRenderer::with_device(device.clone())?;
        let target = MetalRenderTarget::new(
            &device,
            config.width,
            config.height,
            config.color_format,
            config.sample_count,
        )?;
        Ok(Self {
            device,
            renderer,
            target,
            config,
            last_stats: MetalRenderStats::default(),
        })
    }

    /// Draw one frame and read it back as tightly-packed RGBA8, top row first.
    pub fn render_to_rgba(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
    ) -> Result<Vec<u8>, MetalError> {
        self.last_stats = self
            .renderer
            .render(scene, camera, &self.target.attachments())?;
        self.target.read_rgba(&self.device)
    }

    /// Draw one frame, leaving the result on the GPU. Pair with
    /// [`target`](Self::target) to read it back or sample it.
    pub fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
    ) -> Result<MetalRenderStats, MetalError> {
        self.last_stats = self
            .renderer
            .render(scene, camera, &self.target.attachments())?;
        Ok(self.last_stats)
    }

    /// Reallocate the target at a new size. Caches survive — geometry and
    /// textures are not attachment-dependent.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), MetalError> {
        if (width.max(1), height.max(1)) == self.target.size() {
            return Ok(());
        }
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.target = MetalRenderTarget::new(
            &self.device,
            self.config.width,
            self.config.height,
            self.config.color_format,
            self.config.sample_count,
        )?;
        Ok(())
    }

    /// Output size in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.target.size()
    }

    /// Statistics from the most recent frame.
    pub fn stats(&self) -> MetalRenderStats {
        self.last_stats
    }

    /// The underlying device.
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// The underlying renderer, for cache control.
    pub fn renderer_mut(&mut self) -> &mut MetalRenderer {
        &mut self.renderer
    }

    /// The offscreen target.
    pub fn target(&self) -> &MetalRenderTarget {
        &self.target
    }
}

impl std::fmt::Debug for MetalHeadlessRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalHeadlessRenderer")
            .field("device", &self.device.name())
            .field("size", &self.target.size())
            .field("sample_count", &self.target.sample_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_carries_its_settings() {
        let b = MetalHeadlessBuilder::default()
            .size(320, 240)
            .msaa(4)
            .color_format(pixel_format::BGRA8_UNORM_SRGB);
        assert_eq!(b.config.width, 320);
        assert_eq!(b.config.height, 240);
        assert_eq!(b.config.sample_count, 4);
        assert_eq!(b.config.color_format, pixel_format::BGRA8_UNORM_SRGB);
    }

    #[test]
    fn msaa_never_goes_below_one() {
        assert_eq!(
            MetalHeadlessBuilder::default().msaa(0).config.sample_count,
            1
        );
    }
}
