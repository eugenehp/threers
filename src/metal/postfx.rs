//! GPU bloom, supersample downsample, and bilinear upscale for Metal.
//!
//! Metal counterparts of [`crate::renderer::bloom::Bloom`] and
//! [`crate::renderer::downsample::Downsampler`]. Compose them with
//! [`PostFxChain`] on a finished colour target before readback.

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg3, sel, AutoreleasePool, Id, Owned,
    NIL,
};
use super::resources::new_texture_2d;

const BLOOM_SHRINK: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParams {
    threshold: f32,
    radius: f32,
    dx: f32,
    dy: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleParams {
    factor: u32,
    srgb: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapParams {
    exposure: f32,
    white: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    near_z: f32,
    far_z: f32,
    radius: f32,
    strength: f32,
    tan_half_fov: f32,
    aspect: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Camera shape SSAO needs to reconstruct view space, plus its two dials.
/// `near`/`far` MUST match the camera the depth was drawn with — the
/// linearisation is wrong otherwise, and wrong quietly.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SsaoSettings {
    pub strength: f32,
    /// World units, so it tracks scene scale rather than resolution.
    pub radius: f32,
    pub near: f32,
    pub far: f32,
    pub tan_half_fov: f32,
    pub aspect: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UpscaleParams {
    src_size: [f32; 2],
    dst_size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    strength: f32,
    bloom_size: [f32; 2],
    /// Match fly `ref_look` chroma punch (1 = identity).
    chroma_sat: f32,
    chroma_lift: f32,
}

/// Depth-only ambient occlusion as a fullscreen pass: colour in, colour times
/// occlusion out, at the source resolution.
///
/// Full resolution rather than the half-res the wgpu pass uses. That pass runs
/// on surfaces metres across, where AO is genuinely low-frequency; here the
/// occluders are neurites a pixel or two wide, and halving the resolution
/// throws away the very contacts the effect exists to show.
pub struct MetalSsao {
    texture: Owned,
    params: Owned,
    width: u32,
    height: u32,
    settings: SsaoSettings,
}

impl MetalSsao {
    pub fn new(
        device: &MetalDevice,
        width: u32,
        height: u32,
        color_format: NSUInteger,
        settings: SsaoSettings,
    ) -> Result<Self, MetalError> {
        let texture = new_texture_2d(
            device,
            width,
            height,
            color_format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        Ok(Self {
            texture,
            params: param_buffer(
                device,
                SsaoParams {
                    near_z: settings.near,
                    far_z: settings.far,
                    radius: settings.radius,
                    strength: settings.strength,
                    tan_half_fov: settings.tan_half_fov,
                    aspect: settings.aspect,
                    _pad0: 0.0,
                    _pad1: 0.0,
                },
            )?,
            width,
            height,
            settings,
        })
    }

    pub fn texture(&self) -> Id {
        self.texture.id()
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn settings(&self) -> SsaoSettings {
        self.settings
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn encode(
        &self,
        pipelines: &PostFxPipelines,
        cmd: Id,
        src: Id,
        depth: Id,
        color_format: NSUInteger,
    ) -> Result<(), MetalError> {
        let Some(pipeline) = pipelines.ssao_for(color_format) else {
            return Ok(());
        };
        encode_fullscreen(
            cmd,
            pipeline,
            self.texture.id(),
            color_format,
            |enc| unsafe {
                let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), src, 0usize);
                let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), depth, 1usize);
                let _: () = msg3(
                    enc,
                    sel!("setFragmentBuffer:offset:atIndex:"),
                    self.params.id(),
                    0usize,
                    0usize,
                );
            },
        )
    }
}

/// Cached post-processing pipeline states for a device.
pub struct PostFxPipelines {
    bloom_threshold: Owned,
    bloom_blur: Owned,
    downsample: Owned,
    upscale: Owned,
    composite: Owned,
    /// HDR variants. An HDR chain keeps every intermediate in float, so the
    /// passes that write those intermediates need pipelines declared against
    /// the float format — a pipeline's colour format has to match the
    /// attachment it renders into.
    downsample_hdr: Owned,
    composite_hdr: Owned,
    ssao_hdr: Option<Owned>,
    /// Float in, sRGB out. The step that turns an HDR chain into a picture.
    tonemap: Owned,
    /// Optional, unlike its five neighbours: the chain is built with
    /// `PostFxChain::new(..).ok()`, so a pipeline that fails to build takes
    /// every other pass down with it. AO is the one pass here that is a
    /// nice-to-have, and `depth2d` is the one feature that might not build, so
    /// it degrades to "no AO" rather than "no post-processing".
    ssao: Option<Owned>,
}

/// True for the float format the HDR chain uses.
fn is_hdr(format: NSUInteger) -> bool {
    format == pixel_format::RGBA16_FLOAT
}

impl PostFxPipelines {
    fn downsample_for(&self, format: NSUInteger) -> Id {
        if is_hdr(format) { self.downsample_hdr.id() } else { self.downsample.id() }
    }

    fn composite_for(&self, format: NSUInteger) -> Id {
        if is_hdr(format) { self.composite_hdr.id() } else { self.composite.id() }
    }

    fn ssao_for(&self, format: NSUInteger) -> Option<Id> {
        if is_hdr(format) {
            self.ssao_hdr.as_ref().map(|p| p.id())
        } else {
            self.ssao.as_ref().map(|p| p.id())
        }
    }

    pub fn new(device: &MetalDevice) -> Result<Self, MetalError> {
        Ok(Self {
            // Bloom chain stays in linear float16 (matches wgpu headless bloom).
            bloom_threshold: fullscreen_pipeline(
                device,
                "fs_bloom_threshold",
                pixel_format::RGBA16_FLOAT,
            )?,
            bloom_blur: fullscreen_pipeline(
                device,
                "fs_bloom_blur",
                pixel_format::RGBA16_FLOAT,
            )?,
            downsample: fullscreen_pipeline(
                device,
                "fs_downsample",
                pixel_format::RGBA8_UNORM_SRGB,
            )?,
            upscale: fullscreen_pipeline(
                device,
                "fs_upscale",
                pixel_format::RGBA8_UNORM_SRGB,
            )?,
            composite: fullscreen_pipeline(
                device,
                "fs_bloom_composite",
                pixel_format::RGBA8_UNORM_SRGB,
            )?,
            ssao: match fullscreen_pipeline(device, "fs_ssao", pixel_format::RGBA8_UNORM_SRGB) {
                Ok(p) => Some(p),
                Err(e) => {
                    log::warn!("SSAO pipeline unavailable, continuing without it: {e}");
                    None
                }
            },
            downsample_hdr: fullscreen_pipeline(
                device,
                "fs_downsample",
                pixel_format::RGBA16_FLOAT,
            )?,
            composite_hdr: fullscreen_pipeline(
                device,
                "fs_bloom_composite",
                pixel_format::RGBA16_FLOAT,
            )?,
            ssao_hdr: fullscreen_pipeline(device, "fs_ssao", pixel_format::RGBA16_FLOAT).ok(),
            tonemap: fullscreen_pipeline(device, "fs_tonemap", pixel_format::RGBA8_UNORM_SRGB)?,
        })
    }
}

/// Threshold + quarter-res blur, matching the wgpu headless bloom.
pub struct MetalBloom {
    tex_a: Owned,
    tex_b: Owned,
    bw: u32,
    bh: u32,
    src_size: (u32, u32),
    threshold_buf: Owned,
    blur_h_buf: Owned,
    blur_v_buf: Owned,
}

impl MetalBloom {
    pub fn new(
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        threshold: f32,
        radius: f32,
    ) -> Result<Self, MetalError> {
        let (bw, bh) = ((src_w / BLOOM_SHRINK).max(1), (src_h / BLOOM_SHRINK).max(1));
        let tex_a = new_texture_2d(
            device,
            bw,
            bh,
            pixel_format::RGBA16_FLOAT,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        let tex_b = new_texture_2d(
            device,
            bw,
            bh,
            pixel_format::RGBA16_FLOAT,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        Ok(Self {
            tex_a,
            tex_b,
            bw,
            bh,
            src_size: (src_w, src_h),
            threshold_buf: param_buffer(
                device,
                BloomParams {
                    threshold,
                    radius,
                    dx: 0.0,
                    dy: 0.0,
                },
            )?,
            blur_h_buf: param_buffer(
                device,
                BloomParams {
                    threshold,
                    radius,
                    dx: 1.0,
                    dy: 0.0,
                },
            )?,
            blur_v_buf: param_buffer(
                device,
                BloomParams {
                    threshold,
                    radius,
                    dx: 0.0,
                    dy: 1.0,
                },
            )?,
        })
    }

    pub fn src_size(&self) -> (u32, u32) {
        self.src_size
    }

    pub fn texture(&self) -> Id {
        self.tex_a.id()
    }

    pub fn encode(
        &self,
        pipelines: &PostFxPipelines,
        cmd: Id,
        src: Id,
        _color_format: NSUInteger,
    ) -> Result<(), MetalError> {
        let bloom_fmt = pixel_format::RGBA16_FLOAT;
        let pass = |dst: Id, pipeline: Id, input: Id, params: &Owned| -> Result<(), MetalError> {
            encode_fullscreen(cmd, pipeline, dst, bloom_fmt, |enc| {
                unsafe {
                    let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), input, 0usize);
                    let _: () = msg3(
                        enc,
                        sel!("setFragmentBuffer:offset:atIndex:"),
                        params.id(),
                        0usize,
                        0usize,
                    );
                }
            })
        };
        pass(
            self.tex_a.id(),
            pipelines.bloom_threshold.id(),
            src,
            &self.threshold_buf,
        )?;
        pass(
            self.tex_b.id(),
            pipelines.bloom_blur.id(),
            self.tex_a.id(),
            &self.blur_h_buf,
        )?;
        pass(
            self.tex_a.id(),
            pipelines.bloom_blur.id(),
            self.tex_b.id(),
            &self.blur_v_buf,
        )
    }
}

/// Block-average a render target down by `factor`, in linear light when `srgb`.
pub struct MetalDownsampler {
    texture: Owned,
    factor: u32,
    width: u32,
    height: u32,
    params: Owned,
}

impl MetalDownsampler {
    pub fn new(
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        factor: u32,
        color_format: NSUInteger,
    ) -> Result<Self, MetalError> {
        if factor < 2 || !src_w.is_multiple_of(factor) || !src_h.is_multiple_of(factor) {
            return Err(MetalError::Unsupported(format!(
                "downsample factor {factor} invalid for {src_w}x{src_h}"
            )));
        }
        let (width, height) = (src_w / factor, src_h / factor);
        let srgb = u32::from(pixel_format::is_srgb(color_format));
        let texture = new_texture_2d(
            device,
            width,
            height,
            color_format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        Ok(Self {
            texture,
            factor,
            width,
            height,
            params: param_buffer(
                device,
                DownsampleParams {
                    factor,
                    srgb,
                    _pad0: 0,
                    _pad1: 0,
                },
            )?,
        })
    }

    pub fn texture(&self) -> Id {
        self.texture.id()
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn factor(&self) -> u32 {
        self.factor
    }

    // `Id` is `*mut Object`, so clippy asks for `unsafe fn`. Every `Id` in this
    // module comes from the Metal runtime and is only ever handed back to it;
    // making the encoders unsafe would put the keyword on the whole backend
    // without making any caller check anything it is not already checking.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn encode(
        &self,
        pipelines: &PostFxPipelines,
        cmd: Id,
        src: Id,
        color_format: NSUInteger,
    ) -> Result<(), MetalError> {
        encode_fullscreen(
            cmd,
            pipelines.downsample_for(color_format),
            self.texture.id(),
            color_format,
            |enc| {
                unsafe {
                    let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), src, 0usize);
                    let _: () = msg3(
                        enc,
                        sel!("setFragmentBuffer:offset:atIndex:"),
                        self.params.id(),
                        0usize,
                        0usize,
                    );
                }
            },
        )
    }
}

/// Bilinear upscale from a smaller texture to a larger render target.
pub struct MetalUpscaler {
    texture: Owned,
    pub(crate) src_size: (u32, u32),
    pub(crate) dst_size: (u32, u32),
    params: Owned,
}

impl MetalUpscaler {
    pub fn new(
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        color_format: NSUInteger,
    ) -> Result<Self, MetalError> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Err(MetalError::Unsupported("upscale sizes must be non-zero".into()));
        }
        let texture = new_texture_2d(
            device,
            dst_w,
            dst_h,
            color_format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        Ok(Self {
            texture,
            src_size: (src_w, src_h),
            dst_size: (dst_w, dst_h),
            params: param_buffer(
                device,
                UpscaleParams {
                    src_size: [src_w as f32, src_h as f32],
                    dst_size: [dst_w as f32, dst_h as f32],
                },
            )?,
        })
    }

    pub fn texture(&self) -> Id {
        self.texture.id()
    }

    pub fn dst_size(&self) -> (u32, u32) {
        self.dst_size
    }

    // `Id` is `*mut Object`, so clippy asks for `unsafe fn`. Every `Id` in this
    // module comes from the Metal runtime and is only ever handed back to it;
    // making the encoders unsafe would put the keyword on the whole backend
    // without making any caller check anything it is not already checking.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn encode(
        &self,
        pipelines: &PostFxPipelines,
        cmd: Id,
        src: Id,
        color_format: NSUInteger,
    ) -> Result<(), MetalError> {
        encode_fullscreen(
            cmd,
            pipelines.upscale.id(),
            self.texture.id(),
            color_format,
            |enc| {
                unsafe {
                    let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), src, 0usize);
                    let _: () = msg3(
                        enc,
                        sel!("setFragmentBuffer:offset:atIndex:"),
                        self.params.id(),
                        0usize,
                        0usize,
                    );
                }
            },
        )
    }
}

/// Owns postfx resources and encodes bloom / downsample / upscale passes.
pub struct PostFxChain {
    pipelines: PostFxPipelines,
    bloom: Option<MetalBloom>,
    downsample: Option<MetalDownsampler>,
    upscale: Option<MetalUpscaler>,
    ssao: Option<MetalSsao>,
    scratch: Option<Owned>,
    /// LDR destination for the tone map, and its parameters.
    tonemapped: Option<Owned>,
    tonemap_params: Option<Owned>,
    exposure: f32,
    white: f32,
    composite_params: Option<Owned>,
    bloom_strength: f32,
    chroma_sat: f32,
    chroma_lift: f32,
}

impl PostFxChain {
    pub fn new(device: &MetalDevice) -> Result<Self, MetalError> {
        Ok(Self {
            pipelines: PostFxPipelines::new(device)?,
            bloom: None,
            downsample: None,
            upscale: None,
            ssao: None,
            scratch: None,
            tonemapped: None,
            tonemap_params: None,
            exposure: 1.0,
            white: 0.0,
            composite_params: None,
            bloom_strength: 0.0,
            chroma_sat: 1.0,
            chroma_lift: 0.0,
        })
    }

    /// Drop cached pass resources after a resize or bloom-parameter change.
    pub fn invalidate(&mut self) {
        self.bloom = None;
        self.downsample = None;
        self.upscale = None;
        self.ssao = None;
        self.scratch = None;
        self.tonemapped = None;
        self.composite_params = None;
    }

    /// Configure bloom. Rebuilds GPU resources when size or parameters change.
    pub fn set_bloom(
        &mut self,
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        strength: f32,
        threshold: f32,
        radius: f32,
    ) -> Result<(), MetalError> {
        if strength <= 0.0 {
            self.bloom = None;
            self.bloom_strength = 0.0;
            return Ok(());
        }
        let rebuild = self
            .bloom
            .as_ref()
            .map(|b| b.src_size() != (src_w, src_h))
            .unwrap_or(true);
        if rebuild {
            self.bloom = Some(MetalBloom::new(device, src_w, src_h, threshold, radius)?);
        }
        self.bloom_strength = strength;
        Ok(())
    }

    /// GPU chroma punch applied in the bloom composite (1/0 = identity).
    pub fn set_chroma(&mut self, sat_scale: f32, lift: f32) {
        self.chroma_sat = sat_scale.max(0.0);
        self.chroma_lift = lift.max(0.0);
    }

    pub fn ensure_downsample(
        &mut self,
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        factor: u32,
        format: NSUInteger,
    ) -> Result<(), MetalError> {
        if factor < 2 {
            self.downsample = None;
            return Ok(());
        }
        let rebuild = self
            .downsample
            .as_ref()
            .map(|d| d.factor() != factor || d.size() != (src_w / factor, src_h / factor))
            .unwrap_or(true);
        if rebuild {
            self.downsample = Some(MetalDownsampler::new(device, src_w, src_h, factor, format)?);
        }
        Ok(())
    }

    /// Exposure applied before the filmic curve, and the luminance that maps to
    /// white (0 = use the curve's own shoulder).
    pub fn set_tonemap(&mut self, exposure: f32, white: f32) {
        if (self.exposure, self.white) != (exposure, white) {
            self.exposure = exposure;
            self.white = white;
            self.tonemap_params = None;
        }
    }

    /// Configure SSAO, or clear it when `settings` is `None`. Rebuilds only
    /// when the size or a parameter actually changed — a turntable that keeps
    /// the same near/far reuses the buffer for the whole clip.
    pub fn ensure_ssao(
        &mut self,
        device: &MetalDevice,
        width: u32,
        height: u32,
        format: NSUInteger,
        settings: Option<SsaoSettings>,
    ) -> Result<(), MetalError> {
        let Some(settings) = settings
            .filter(|s| s.strength > 0.0)
            .filter(|_| self.pipelines.ssao.is_some() || self.pipelines.ssao_hdr.is_some())
        else {
            self.ssao = None;
            return Ok(());
        };
        let rebuild = self
            .ssao
            .as_ref()
            .map(|a| a.size() != (width, height) || a.settings() != settings)
            .unwrap_or(true);
        if rebuild {
            self.ssao = Some(MetalSsao::new(device, width, height, format, settings)?);
        }
        Ok(())
    }

    pub fn ensure_upscale(
        &mut self,
        device: &MetalDevice,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        format: NSUInteger,
    ) -> Result<(), MetalError> {
        if (src_w, src_h) == (dst_w, dst_h) {
            self.upscale = None;
            return Ok(());
        }
        let rebuild = self
            .upscale
            .as_ref()
            .map(|u| u.src_size != (src_w, src_h) || u.dst_size != (dst_w, dst_h))
            .unwrap_or(true);
        if rebuild {
            self.upscale = Some(MetalUpscaler::new(
                device, src_w, src_h, dst_w, dst_h, format,
            )?);
        }
        Ok(())
    }

    /// Encode postfx onto `cmd` without committing. Returns the colour texture
    /// to read back, or `None` if no postfx was needed (`src` is used as-is).
    pub fn encode_on(
        &mut self,
        device: &MetalDevice,
        cmd: Id,
        src: Id,
        depth: Id,
        src_w: u32,
        src_h: u32,
        format: NSUInteger,
    ) -> Result<Option<Id>, MetalError> {
        // An HDR target *always* needs the chain, whatever else is configured:
        // the readback buffer is 8-bit, so without the tone map the float
        // texture would be blitted into it and read as garbage.
        let needs = is_hdr(format)
            || self.downsample.is_some()
            || self.upscale.is_some()
            || self.ssao.is_some()
            || (self.bloom.is_some() && self.bloom_strength > 0.0);
        if !needs {
            return Ok(None);
        }

        let mut current = src;
        // Tracked because the tone map changes it mid-chain, and a pipeline has
        // to match the attachment it writes.
        let mut cur_format = format;

        // AO first, at full render resolution and before any resolve: it is a
        // lighting term, so everything downstream — bloom especially — should
        // see the darkened image rather than bloom the un-occluded one.
        if let Some(ao) = self.ssao.as_ref() {
            if !depth.is_null() {
                ao.encode(&self.pipelines, cmd, current, depth, format)?;
                current = ao.texture();
            }
        }

        if let Some(ds) = self.downsample.as_ref() {
            ds.encode(&self.pipelines, cmd, current, format)?;
            current = ds.texture();
        }

        // Bloom before upscale so the blur runs at the cheaper resolution.
        if self.bloom_strength > 0.0 {
            if let Some(bloom) = self.bloom.as_ref() {
                bloom.encode(&self.pipelines, cmd, current, format)?;
            }
            if let Some(bloom) = self.bloom.as_ref() {
                let bloom_tex = bloom.texture();
                let bloom_bw = bloom.bw;
                let bloom_bh = bloom.bh;
                let (w, h) = if let Some(ds) = self.downsample.as_ref() {
                    ds.size()
                } else {
                    (src_w, src_h)
                };
                let rebuild = self
                    .scratch
                    .as_ref()
                    .map(|t| {
                        let tw: NSUInteger = unsafe { msg0(t.id(), sel!("width")) };
                        let th: NSUInteger = unsafe { msg0(t.id(), sel!("height")) };
                        tw as u32 != w || th as u32 != h
                    })
                    .unwrap_or(true);
                if rebuild {
                    self.scratch = Some(new_texture_2d(
                        device,
                        w,
                        h,
                        format,
                        texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
                        storage_mode::PRIVATE,
                        1,
                        1,
                    )?);
                }
                let scratch_id = self.scratch.as_ref().expect("scratch").id();
                self.composite_params = Some(param_buffer(
                    device,
                    CompositeParams {
                        strength: self.bloom_strength,
                        bloom_size: [bloom_bw as f32, bloom_bh as f32],
                        chroma_sat: self.chroma_sat,
                        chroma_lift: self.chroma_lift,
                    },
                )?);
                let composite = self.composite_params.as_ref().expect("composite params");
                encode_fullscreen(
                    cmd,
                    self.pipelines.composite_for(format),
                    scratch_id,
                    format,
                    |enc| {
                        unsafe {
                            let _: () =
                                msg2(enc, sel!("setFragmentTexture:atIndex:"), current, 0usize);
                            let _: () =
                                msg2(enc, sel!("setFragmentTexture:atIndex:"), bloom_tex, 1usize);
                            let _: () = msg3(
                                enc,
                                sel!("setFragmentBuffer:offset:atIndex:"),
                                composite.id(),
                                0usize,
                                0usize,
                            );
                        }
                    },
                )?;
                current = scratch_id;
            }
        }

        // Tone map last in float, before any LDR resampling: compressing after
        // an upscale would resample values the display can never show.
        if is_hdr(cur_format) {
            let (w, h) = if let Some(ds) = self.downsample.as_ref() {
                ds.size()
            } else {
                (src_w, src_h)
            };
            let rebuild = self
                .tonemapped
                .as_ref()
                .map(|t| unsafe {
                    let tw: NSUInteger = msg0(t.id(), sel!("width"));
                    let th: NSUInteger = msg0(t.id(), sel!("height"));
                    tw as u32 != w || th as u32 != h
                })
                .unwrap_or(true);
            if rebuild {
                self.tonemapped = Some(new_texture_2d(
                    device,
                    w,
                    h,
                    pixel_format::RGBA8_UNORM_SRGB,
                    texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
                    storage_mode::PRIVATE,
                    1,
                    1,
                )?);
            }
            if self.tonemap_params.is_none() {
                self.tonemap_params = Some(param_buffer(
                    device,
                    TonemapParams {
                        exposure: self.exposure,
                        white: self.white,
                        _pad0: 0.0,
                        _pad1: 0.0,
                    },
                )?);
            }
            let dst = self.tonemapped.as_ref().expect("tonemap target").id();
            let params = self.tonemap_params.as_ref().expect("tonemap params").id();
            encode_fullscreen(
                cmd,
                self.pipelines.tonemap.id(),
                dst,
                pixel_format::RGBA8_UNORM_SRGB,
                |enc| unsafe {
                    let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), current, 0usize);
                    let _: () = msg3(
                        enc,
                        sel!("setFragmentBuffer:offset:atIndex:"),
                        params,
                        0usize,
                        0usize,
                    );
                },
            )?;
            current = dst;
            cur_format = pixel_format::RGBA8_UNORM_SRGB;
        }

        if let Some(up) = self.upscale.as_ref() {
            up.encode(&self.pipelines, cmd, current, cur_format)?;
            current = up.texture();
        }

        Ok(Some(current))
    }

    /// Encode postfx into a new command buffer and **commit without waiting**.
    pub fn encode_async(
        &mut self,
        device: &MetalDevice,
        src: Id,
        depth: Id,
        src_w: u32,
        src_h: u32,
        format: NSUInteger,
    ) -> Result<Option<(Id, Id)>, MetalError> {
        let _pool = AutoreleasePool::new();
        let needs = self.downsample.is_some()
            || self.upscale.is_some()
            || self.ssao.is_some()
            || (self.bloom.is_some() && self.bloom_strength > 0.0);
        if !needs {
            return Ok(None);
        }
        let cmd = unsafe { device.command_buffer() };
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let tex = self
            .encode_on(device, cmd, src, depth, src_w, src_h, format)?
            .expect("needs was true");
        unsafe {
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(Some((cmd, tex)))
    }

    /// Returns the texture to read back after post-processing (`src` if unchanged).
    pub fn encode(
        &mut self,
        device: &MetalDevice,
        src: Id,
        depth: Id,
        src_w: u32,
        src_h: u32,
        format: NSUInteger,
    ) -> Result<Id, MetalError> {
        match self.encode_async(device, src, depth, src_w, src_h, format)? {
            None => Ok(src),
            Some((cmd, tex)) => {
                unsafe {
                    let _: () = msg0(cmd, sel!("waitUntilCompleted"));
                }
                Ok(tex)
            }
        }
    }
}

fn param_buffer<T: Copy + bytemuck::Pod>(device: &MetalDevice, value: T) -> Result<Owned, MetalError> {
    device.new_buffer(bytemuck::bytes_of(&value))
}

fn fullscreen_pipeline(
    device: &MetalDevice,
    fragment: &str,
    color_format: NSUInteger,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let vs = device.function("vs_fullscreen")?;
    let fs = device.function(fragment)?;
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
        let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
        let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setPixelFormat:"), color_format);
        let mut err: Id = NIL;
        let state: Id = msg2(
            device.device_id(),
            sel!("newRenderPipelineStateWithDescriptor:error:"),
            desc.id(),
            &mut err as *mut Id,
        );
        Owned::from_retained(state).ok_or_else(|| MetalError::PipelineCreation(error_message(err)))
    }
}

fn encode_fullscreen(
    cmd: Id,
    pipeline: Id,
    dst: Id,
    _color_format: NSUInteger,
    bind: impl FnOnce(Id),
) -> Result<(), MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = render_pass_descriptor(dst);
        let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
        if enc.is_null() {
            return Err(MetalError::Unsupported(
                "fullscreen postfx render encoder failed".into(),
            ));
        }
        let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
        bind(enc);
        let _: () = msg3(
            enc,
            sel!("drawPrimitives:vertexStart:vertexCount:"),
            primitive::TRIANGLE,
            0usize,
            3usize,
        );
        let _: () = msg0(enc, sel!("endEncoding"));
    }
    Ok(())
}

fn render_pass_descriptor(dst: Id) -> Id {
    unsafe {
        let desc: Id = msg0(
            class!("MTLRenderPassDescriptor"),
            sel!("renderPassDescriptor"),
        );
        let attachments: Id = msg0(desc, sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setTexture:"), dst);
        let _: () = msg1(color, sel!("setLoadAction:"), load_action::CLEAR);
        let _: () = msg1(color, sel!("setStoreAction:"), store_action::STORE);
        let clear = MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        };
        let _: () = msg1(color, sel!("setClearColor:"), clear);
        desc
    }
}

#[cfg(all(test, feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod gpu_tests {
    use super::*;
    use crate::metal::objc::AutoreleasePool;

    #[test]
    fn downsample_pass_completes() {
        let device = match MetalDevice::new() {
            Ok(d) => d,
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let _pool = AutoreleasePool::new();
        let format = pixel_format::RGBA8_UNORM_SRGB;
        let src = new_texture_2d(
            &device,
            64,
            64,
            format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )
        .unwrap();
        let ds = MetalDownsampler::new(&device, 64, 64, 2, format).unwrap();
        let pipelines = PostFxPipelines::new(&device).unwrap();
        let cmd = unsafe { device.command_buffer() };
        assert!(!cmd.is_null());
        ds.encode(&pipelines, cmd, src.id(), format).unwrap();
        unsafe {
            let _: () = msg0(cmd, sel!("commit"));
            let _: () = msg0(cmd, sel!("waitUntilCompleted"));
        }
    }
}
