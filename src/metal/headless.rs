//! Batteries-included offscreen rendering with Metal.
//!
//! The Metal counterpart of [`crate::renderer::HeadlessRenderer`]: owns the
//! device, the renderer and an offscreen target, and hands back RGBA bytes.

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, sel, AutoreleasePool, Owned};
use super::postfx::{PostFxChain, SsaoSettings};
use super::renderer::{MetalRenderStats, MetalRenderer, RenderView};
use super::target::MetalRenderTarget;
use crate::cameras::Camera;
use crate::scene::Scene;

/// A GPU readback in flight (Metal blit committed, not yet waited on).
struct PendingMetalReadback {
    /// Retained so it outlives the enqueue autorelease pool.
    cmd: Owned,
    staging: Owned,
    width: u32,
    height: u32,
    /// When true, unpack via the render target colour attachment layout.
    from_target: bool,
}

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
    /// Render at `width × height × supersample / render_scale` and resolve on
    /// the GPU before readback (defaults: 1, 1 = native).
    pub supersample: u32,
    pub render_scale: u32,
}

impl Default for MetalHeadlessConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            sample_count: 1,
            color_format: pixel_format::RGBA8_UNORM_SRGB,
            supersample: 1,
            render_scale: 1,
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

    /// Supersample factor (render larger, GPU-downsample on readback).
    pub fn supersample(mut self, factor: u32) -> Self {
        self.config.supersample = factor.max(1);
        self
    }

    /// Render at `1/N` resolution and GPU-upscale on readback.
    pub fn render_scale(mut self, scale: u32) -> Self {
        self.config.render_scale = scale.max(1);
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
    export_w: u32,
    export_h: u32,
    postfx: Option<PostFxChain>,
    ssao_cfg: Option<SsaoSettings>,
    bloom_cfg: Option<(f32, f32, f32)>,
    last_stats: MetalRenderStats,
    /// Staging buffers recycled across frames.
    staging_pool: Vec<Owned>,
    /// Previous frame's readback, returned on the next call.
    pending: Option<PendingMetalReadback>,
    staging_bytes: usize,
}

impl MetalHeadlessRenderer {
    /// Start a fluent builder.
    pub fn builder() -> MetalHeadlessBuilder {
        MetalHeadlessBuilder::default()
    }

    /// Construct from a config, acquiring the GPU.
    pub fn new(config: MetalHeadlessConfig) -> Result<Self, MetalError> {
        let device = MetalDevice::new()?;
        let postfx = PostFxChain::new(&device).ok();
        let renderer = MetalRenderer::with_device(device.clone())?;
        let (export_w, export_h) = (config.width.max(1), config.height.max(1));
        let (render_w, render_h) = render_dimensions(
            export_w,
            export_h,
            config.supersample,
            config.render_scale,
        );
        let target = MetalRenderTarget::new(
            &device,
            render_w,
            render_h,
            config.color_format,
            config.sample_count,
        )?;
        log::info!(
            "metal headless: export {export_w}×{export_h}, render target {render_w}×{render_h} \
             (ss {}, scale {}), postfx {}",
            config.supersample.max(1),
            config.render_scale.max(1),
            if postfx.is_some() { "ready" } else { "UNAVAILABLE" }
        );
        Ok(Self {
            device,
            renderer,
            target,
            config,
            export_w,
            export_h,
            postfx,
            ssao_cfg: None,
            bloom_cfg: None,
            last_stats: MetalRenderStats::default(),
            staging_pool: Vec::new(),
            pending: None,
            staging_bytes: 0,
        })
    }

    fn needs_postfx(&self) -> bool {
        // An HDR target must run the chain to be tone mapped down to the 8-bit
        // readback, whatever else is off.
        self.config.color_format == pixel_format::RGBA16_FLOAT
            || self.config.supersample > 1
            || self.config.render_scale > 1
            || self.ssao_cfg.is_some_and(|s| s.strength > 0.0)
            || self
                .bloom_cfg
                .is_some_and(|(s, _, _)| s > 0.0)
    }

    fn staging_buffer(&mut self) -> Result<Owned, MetalError> {
        let (width, height) = if self.needs_postfx() {
            (self.export_w, self.export_h)
        } else {
            self.target.size()
        };
        let padded = ((width as usize * 4).div_ceil(256)) * 256;
        let need = padded * height as usize;
        if need != self.staging_bytes {
            self.staging_pool.clear();
            self.staging_bytes = need;
        }
        if let Some(buf) = self.staging_pool.pop() {
            return Ok(buf);
        }
        self.device.new_buffer(&vec![0u8; need])
    }

    fn recycle_staging(&mut self, buf: Owned) {
        self.staging_pool.push(buf);
    }

    /// GPU bloom on readback: `strength` scales the add, `threshold` is linear
    /// luminance, `radius` widens the blur (in quarter-res texels).
    pub fn set_bloom(&mut self, strength: f32, threshold: f32, radius: f32) {
        let rebuild = self
            .bloom_cfg
            .map(|(_, t, r)| (t, r) != (threshold, radius))
            .unwrap_or(true);
        self.bloom_cfg = Some((strength, threshold, radius));
        if rebuild {
            if let Some(p) = self.postfx.as_mut() {
                p.invalidate();
            }
        }
    }

    /// Depth-only ambient occlusion. `None` (or zero strength) disables it.
    ///
    /// `near`/`far`/`tan_half_fov`/`aspect` must describe the camera the frame
    /// is drawn with, so a shot that reframes itself should call this each
    /// frame; stale values do not fail loudly, they just slide the occlusion
    /// off the creases.
    pub fn set_ssao(&mut self, settings: Option<SsaoSettings>) {
        // The pass samples the depth attachment after the fact, so the renderer
        // has to be told to keep it. Without this the texture is discarded and
        // reads back as the clear value.
        self.renderer.set_store_depth(settings.is_some());
        self.ssao_cfg = settings;
    }

    /// Draw lines as coverage-weighted quads of this pixel width; 0 = hardware
    /// lines.
    pub fn set_line_width(&mut self, px: f32) {
        self.renderer.set_line_width(px);
    }

    /// Exposure and white point for the HDR tone map. Inert on an LDR target.
    pub fn set_tonemap(&mut self, exposure: f32, white: f32) {
        if let Some(p) = self.postfx.as_mut() {
            p.set_tonemap(exposure, white);
        }
    }

    /// GPU chroma punch in the bloom composite (replaces CPU full-frame grade).
    pub fn set_chroma(&mut self, sat_scale: f32, lift: f32) {
        if let Some(p) = self.postfx.as_mut() {
            p.set_chroma(sat_scale, lift);
        }
    }

    fn queue_postfx_readback(&mut self) -> Result<PendingMetalReadback, MetalError> {
        let (rw, rh) = self.target.size();
        let format = self.config.color_format;
        let (ew, eh) = (self.export_w, self.export_h);
        let staging = self.staging_buffer()?;
        let padded = ((ew as usize * 4).div_ceil(256)) * 256;

        let Some(postfx) = self.postfx.as_mut() else {
            let cmd = self.target.queue_readback(&self.device, &staging)?;
            // `queue_readback` hands back an owned reference already.
            return Ok(PendingMetalReadback {
                cmd,
                staging,
                width: ew,
                height: eh,
                from_target: true,
            });
        };

        // Supersampling already resolves rw×rh down to rw/ss × rh/ss, so the
        // upscale must be judged against that, not the raw target size.
        // Comparing (rw, rh) meant a supersampled render also queued an upscale
        // configured for a source that no longer existed — which both rescaled
        // wrongly and flipped the frame, since fs_upscale samples by `uv`
        // (origin bottom) while the resolve samples by `position` (origin top).
        let (dw, dh) = if self.config.supersample > 1 {
            let ss = self.config.supersample.max(1);
            postfx.ensure_downsample(&self.device, rw, rh, ss, format)?;
            (rw / ss, rh / ss)
        } else {
            (rw, rh)
        };
        if (dw, dh) != (ew, eh) {
            postfx.ensure_upscale(&self.device, dw, dh, ew, eh, format)?;
        }
        // MSAA depth is a `texture2d_ms` and will not bind to the pass's
        // `depth2d`, so AO is skipped rather than wrong when MSAA is on.
        // Supersampling is the sharper choice here anyway (+8.4 dB measured).
        let ssao_ok = self.target.sample_count() == 1;
        if self.ssao_cfg.is_some_and(|s| s.strength > 0.0) && !ssao_ok {
            log::warn!("SSAO skipped: needs a single-sampled depth target (MSAA is on)");
        }
        postfx.ensure_ssao(
            &self.device,
            rw,
            rh,
            format,
            self.ssao_cfg.filter(|_| ssao_ok),
        )?;
        if let Some((strength, threshold, radius)) = self.bloom_cfg {
            if strength > 0.0 {
                let (bw, bh) = if self.config.supersample > 1 {
                    (
                        rw / self.config.supersample.max(1),
                        rh / self.config.supersample.max(1),
                    )
                } else {
                    (rw, rh)
                };
                postfx.set_bloom(&self.device, bw, bh, strength, threshold, radius)?;
            }
        }

        let src = self.target.color_texture();
        // Keep the pool live only while building encoders; retain the CB after.
        let cmd = {
            let _pool = AutoreleasePool::new();
            let cmd = unsafe { self.device.command_buffer() };
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let depth = if ssao_ok {
                self.target.depth_texture()
            } else {
                std::ptr::null_mut()
            };
            match postfx.encode_on(&self.device, cmd, src, depth, rw, rh, format)? {
                None => {
                    let cmd = self.target.queue_readback(&self.device, &staging)?;
                    return Ok(PendingMetalReadback {
                        cmd,
                        staging,
                        width: ew,
                        height: eh,
                        from_target: true,
                    });
                }
                Some(tex) => {
                    crate::metal::target::blit_texture_to_buffer_on_cmd(
                        cmd, tex, ew, eh, &staging, padded,
                    )?;
                    unsafe {
                        let _: () = msg0(cmd, sel!("commit"));
                    }
                    unsafe { Owned::retain(cmd) }.ok_or(MetalError::NoCommandQueue)?
                }
            }
        };
        Ok(PendingMetalReadback {
            cmd,
            staging,
            width: ew,
            height: eh,
            from_target: false,
        })
    }

    fn run_postfx_and_readback(&mut self, out: &mut Vec<u8>) -> Result<(), MetalError> {
        let pending = self.queue_postfx_readback()?;
        self.finish_pending_into(pending, out)
    }

    /// Format of the texture that actually reaches the staging buffer.
    fn readback_format(&self) -> NSUInteger {
        if self.config.color_format == pixel_format::RGBA16_FLOAT {
            pixel_format::RGBA8_UNORM_SRGB
        } else {
            self.config.color_format
        }
    }

    fn finish_pending_into(
        &mut self,
        pending: PendingMetalReadback,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        // The chain's OUTPUT format, which is not the target's when the target
        // is HDR: the tone map has already converted to 8-bit sRGB by the time
        // anything is read back.
        let format = self.readback_format();
        if pending.from_target {
            self.target
                .finish_readback_into(pending.cmd.id(), &pending.staging, out)?;
        } else {
            crate::metal::target::finish_texture_readback_into(
                pending.cmd.id(),
                &pending.staging,
                pending.width,
                pending.height,
                format,
                out,
            )?;
        }
        self.recycle_staging(pending.staging);
        Ok(())
    }

    /// Draw one frame and read it back as tightly-packed RGBA8, top row first.
    pub fn render_to_rgba(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
    ) -> Result<Vec<u8>, MetalError> {
        let mut out = Vec::new();
        self.render_to_rgba_into(scene, camera, &mut out)?;
        Ok(out)
    }

    /// Render and unpack into `out`, reusing pooled staging and `out`'s capacity.
    pub fn render_to_rgba_into(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        self.render_into(scene, camera, out)
    }

    /// Pipelined export: render now, return the *previous* frame's pixels.
    ///
    /// Overlaps GPU draw (+ bloom) with CPU unpack of the last frame. Works
    /// with postfx — previously postfx forced a fully synchronous path.
    pub fn render_to_rgba_pipelined(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        out: &mut Vec<u8>,
    ) -> Result<Option<()>, MetalError> {
        let prev = self.drain_pending_into(out)?;
        self.last_stats = self
            .renderer
            .render(scene, camera, &self.target.attachments())?;
        self.pending = Some(self.queue_postfx_readback()?);
        if prev {
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    /// Drain the last queued readback (end of a pipelined sequence).
    pub fn finish_readback_into(&mut self, out: &mut Vec<u8>) -> Result<bool, MetalError> {
        self.drain_pending_into(out)
    }

    fn drain_pending_into(&mut self, out: &mut Vec<u8>) -> Result<bool, MetalError> {
        let Some(pending) = self.pending.take() else {
            return Ok(false);
        };
        self.finish_pending_into(pending, out)?;
        Ok(true)
    }

    /// Draw one frame for each view and read back as tightly-packed RGBA8.
    pub fn render_views_into(
        &mut self,
        scene: &mut Scene,
        views: &[RenderView],
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        self.last_stats = self
            .renderer
            .render_views(scene, views, &self.target.attachments())?;
        self.run_postfx_and_readback(out)
    }

    /// Pipelined multi-view export: render now, return the *previous* frame.
    pub fn render_views_pipelined(
        &mut self,
        scene: &mut Scene,
        views: &[RenderView],
        out: &mut Vec<u8>,
    ) -> Result<Option<()>, MetalError> {
        let prev = self.drain_pending_into(out)?;
        self.last_stats = self
            .renderer
            .render_views(scene, views, &self.target.attachments())?;
        self.pending = Some(self.queue_postfx_readback()?);
        if prev {
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    /// Optimized read path: pooled staging buffer, reused output `Vec`.
    pub fn render_into(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        self.last_stats = self
            .renderer
            .render(scene, camera, &self.target.attachments())?;
        self.run_postfx_and_readback(out)
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
        let (export_w, export_h) = (width.max(1), height.max(1));
        if (export_w, export_h) == (self.export_w, self.export_h) {
            return Ok(());
        }
        self.export_w = export_w;
        self.export_h = export_h;
        self.config.width = export_w;
        self.config.height = export_h;
        let (render_w, render_h) = render_dimensions(
            export_w,
            export_h,
            self.config.supersample,
            self.config.render_scale,
        );
        self.target = MetalRenderTarget::new(
            &self.device,
            render_w,
            render_h,
            self.config.color_format,
            self.config.sample_count,
        )?;
        if let Some(p) = self.postfx.as_mut() {
            p.invalidate();
        }
        Ok(())
    }

    /// Export/readback size in pixels (after GPU upscale/downsample).
    pub fn export_size(&self) -> (u32, u32) {
        (self.export_w, self.export_h)
    }

    /// Output size in pixels (export resolution).
    pub fn size(&self) -> (u32, u32) {
        (self.export_w, self.export_h)
    }

    /// Internal render target size (before postfx).
    pub fn render_size(&self) -> (u32, u32) {
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

fn render_dimensions(export_w: u32, export_h: u32, supersample: u32, render_scale: u32) -> (u32, u32) {
    let ss = supersample.max(1);
    let rs = render_scale.max(1);
    (
        (export_w * ss).div_ceil(rs).max(1),
        (export_h * ss).div_ceil(rs).max(1),
    )
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
