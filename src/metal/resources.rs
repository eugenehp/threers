//! Textures, samplers and pipeline states.

use std::ffi::c_void;

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg4, sel, AutoreleasePool, Bool, Id,
    Owned, NIL, NO, YES,
};
use crate::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};

/// Everything that distinguishes one `MTLRenderPipelineState` from another.
///
/// Cull mode, winding and fill mode are deliberately absent: Metal sets those
/// on the encoder, so wireframe and two-sided materials cost a state change
/// rather than a pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PipelineKey {
    /// `false` = `vs_mesh`/`fs_mesh`, `true` = `vs_point`/`fs_point`.
    pub point_sprites: bool,
    /// Source-alpha blending, for transparent materials.
    pub blend: bool,
    /// Draw both eyes in one pass, into two slices of an array attachment.
    pub layered: bool,
    /// `MTLPrimitiveTopologyClass`. Left
    /// [`UNSPECIFIED`](crate::metal::enums::topology_class::UNSPECIFIED) so one
    /// pipeline serves every topology — except on the layered path, where a
    /// vertex function that writes `render_target_array_index` must name it.
    pub topology: NSUInteger,
    /// Colour attachment format.
    pub color_format: NSUInteger,
    /// Depth attachment format, or 0 for no depth attachment.
    pub depth_format: NSUInteger,
    /// MSAA sample count (1 = off).
    pub sample_count: u32,
}

/// Build the pipeline state for `key`.
pub fn render_pipeline(device: &MetalDevice, key: PipelineKey) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let (vs, fs) = match (key.point_sprites, key.layered) {
        (false, false) => ("vs_mesh", "fs_mesh"),
        (false, true) => ("vs_mesh_layered", "fs_mesh"),
        (true, false) => ("vs_point", "fs_point"),
        (true, true) => ("vs_point_layered", "fs_point"),
    };
    let vertex_fn = device.function(vs)?;
    let fragment_fn = device.function(fs)?;

    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vertex_fn.id());
        let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fragment_fn.id());
        let _: () = msg1(
            desc.id(),
            sel!("setRasterSampleCount:"),
            key.sample_count.max(1) as NSUInteger,
        );
        if key.depth_format != 0 {
            let _: () = msg1(
                desc.id(),
                sel!("setDepthAttachmentPixelFormat:"),
                key.depth_format,
            );
        }
        if key.topology != topology_class::UNSPECIFIED {
            let _: () = msg1(desc.id(), sel!("setInputPrimitiveTopology:"), key.topology);
        }

        // `colorAttachments[0]` — owned by the descriptor, so no release here.
        let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setPixelFormat:"), key.color_format);
        if key.blend {
            let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
            let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
            let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
            let _: () = msg1(
                color,
                sel!("setSourceRGBBlendFactor:"),
                blend_factor::SOURCE_ALPHA,
            );
            let _: () = msg1(
                color,
                sel!("setDestinationRGBBlendFactor:"),
                blend_factor::ONE_MINUS_SOURCE_ALPHA,
            );
            let _: () = msg1(color, sel!("setSourceAlphaBlendFactor:"), blend_factor::ONE);
            let _: () = msg1(
                color,
                sel!("setDestinationAlphaBlendFactor:"),
                blend_factor::ONE_MINUS_SOURCE_ALPHA,
            );
        } else {
            let _: () = msg1(color, sel!("setBlendingEnabled:"), NO);
        }

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

/// Allocate a 2D texture.
#[allow(clippy::too_many_arguments)]
pub fn new_texture_2d(
    device: &MetalDevice,
    width: u32,
    height: u32,
    pixel_format: NSUInteger,
    usage: NSUInteger,
    storage: NSUInteger,
    sample_count: u32,
    mip_levels: u32,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let ty = if sample_count > 1 {
            texture_type::TYPE_2D_MULTISAMPLE
        } else {
            texture_type::TYPE_2D
        };
        let _: () = msg1(desc.id(), sel!("setTextureType:"), ty);
        let _: () = msg1(desc.id(), sel!("setPixelFormat:"), pixel_format);
        let _: () = msg1(desc.id(), sel!("setWidth:"), width.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setHeight:"), height.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setDepth:"), 1usize);
        let _: () = msg1(
            desc.id(),
            sel!("setMipmapLevelCount:"),
            mip_levels.max(1) as NSUInteger,
        );
        let _: () = msg1(
            desc.id(),
            sel!("setSampleCount:"),
            sample_count.max(1) as NSUInteger,
        );
        let _: () = msg1(desc.id(), sel!("setUsage:"), usage);
        let _: () = msg1(desc.id(), sel!("setStorageMode:"), storage);

        let tex: Id = msg1(
            device.device_id(),
            sel!("newTextureWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(tex).ok_or_else(|| {
            MetalError::Allocation(format!(
                "{width}x{height} texture, format {pixel_format}, {sample_count}x MSAA"
            ))
        })
    }
}

/// Allocate a 2D texture array — the attachment shape stereo rendering draws
/// into, a slice per eye.
pub fn new_texture_array(
    device: &MetalDevice,
    width: u32,
    height: u32,
    pixel_format: NSUInteger,
    usage: NSUInteger,
    slices: u32,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let _: () = msg1(
            desc.id(),
            sel!("setTextureType:"),
            texture_type::TYPE_2D_ARRAY,
        );
        let _: () = msg1(desc.id(), sel!("setPixelFormat:"), pixel_format);
        let _: () = msg1(desc.id(), sel!("setWidth:"), width.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setHeight:"), height.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setDepth:"), 1usize);
        let _: () = msg1(
            desc.id(),
            sel!("setArrayLength:"),
            slices.max(1) as NSUInteger,
        );
        let _: () = msg1(desc.id(), sel!("setMipmapLevelCount:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setSampleCount:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setUsage:"), usage);
        let _: () = msg1(desc.id(), sel!("setStorageMode:"), storage_mode::PRIVATE);
        let tex: Id = msg1(
            device.device_id(),
            sel!("newTextureWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(tex).ok_or_else(|| {
            MetalError::Allocation(format!(
                "{width}x{height} texture array of {slices} slices, format {pixel_format}"
            ))
        })
    }
}

/// The Metal pixel format for a threers texture format, and whether the GPU can
/// filter it.
fn pixel_format_for(format: TextureFormat, bc_supported: bool) -> Result<NSUInteger, MetalError> {
    Ok(match format {
        TextureFormat::Rgba8UnormSrgb => pixel_format::RGBA8_UNORM_SRGB,
        TextureFormat::Rgba8Unorm => pixel_format::RGBA8_UNORM,
        TextureFormat::R8Unorm => pixel_format::R8_UNORM,
        TextureFormat::Rgba16Float => pixel_format::RGBA16_FLOAT,
        TextureFormat::Bc1RgbaUnormSrgb if bc_supported => pixel_format::BC1_RGBA_SRGB,
        TextureFormat::Bc7RgbaUnormSrgb if bc_supported => pixel_format::BC7_RGBA_SRGB,
        other => {
            return Err(MetalError::Unsupported(format!(
                "texture format {other:?} — this GPU reports no BC texture compression"
            )))
        }
    })
}

/// The storage mode for a texture the CPU writes with `replaceRegion:`.
///
/// Shared storage is not available for textures on a discrete GPU — there is no
/// memory both sides can address — so those get Managed, where `replaceRegion:`
/// updates the GPU copy for us. Apple silicon takes Shared and skips the copy.
fn cpu_writable_storage(device: &MetalDevice) -> NSUInteger {
    #[cfg(target_os = "ios")]
    {
        let _ = device;
        storage_mode::SHARED
    }
    #[cfg(not(target_os = "ios"))]
    {
        if device.has_unified_memory() {
            storage_mode::SHARED
        } else {
            storage_mode::MANAGED
        }
    }
}

/// Whether this device can sample BC1/BC7. False on Apple GPUs before M3.
pub fn supports_bc(device: &MetalDevice) -> bool {
    unsafe {
        let responds: Bool = msg1(
            device.device_id(),
            sel!("respondsToSelector:"),
            sel!("supportsBCTextureCompression"),
        );
        if responds == 0 {
            return false;
        }
        let ok: Bool = msg0(device.device_id(), sel!("supportsBCTextureCompression"));
        ok != 0
    }
}

/// Upload a threers [`Texture`] and return the `MTLTexture`.
///
/// `flip_y` is applied here rather than in the shader, matching the wgpu
/// backend: three.js UVs have a bottom-left origin and both GPU APIs sample
/// from the top left, so the rows move once at upload instead of every sample.
pub fn upload_texture(device: &MetalDevice, src: &Texture) -> Result<Owned, MetalError> {
    let bc = src.format.is_block_compressed();
    let pixel_format = pixel_format_for(src.format, supports_bc(device))?;
    if bc && src.flip_y {
        return Err(MetalError::Unsupported(
            "flip_y on a block-compressed texture: rows of 4x4 blocks cannot be reordered \
             per row — pre-flip the source"
                .into(),
        ));
    }

    // A block-compressed chain has to arrive already encoded; everything else
    // can be filled in by the GPU.
    let supplied_mips = src.mips.len() as u32;
    let mip_levels = if supplied_mips > 0 {
        supplied_mips + 1
    } else if bc || src.width <= 1 || src.height <= 1 {
        1
    } else {
        32 - src.width.max(src.height).leading_zeros()
    };

    let tex = new_texture_2d(
        device,
        src.width,
        src.height,
        pixel_format,
        texture_usage::SHADER_READ,
        cpu_writable_storage(device),
        1,
        mip_levels,
    )?;

    let (bw, bh) = src.format.block_dim();
    let bytes_per_block = src.format.bytes_per_block();
    let level0 = flip_rows_if_needed(src);
    upload_level(
        &tex,
        0,
        src.width,
        src.height,
        (bw, bh),
        bytes_per_block,
        &level0,
    )?;

    for (i, level) in src.mips.iter().enumerate() {
        let l = i as u32 + 1;
        let w = (src.width >> l).max(1);
        let h = (src.height >> l).max(1);
        upload_level(&tex, l, w, h, (bw, bh), bytes_per_block, level)?;
    }

    if supplied_mips == 0 && mip_levels > 1 {
        generate_mipmaps(device, &tex);
    }
    Ok(tex)
}

/// `replaceRegion:mipmapLevel:withBytes:bytesPerRow:` for one level.
fn upload_level(
    tex: &Owned,
    level: u32,
    width: u32,
    height: u32,
    block: (u32, u32),
    bytes_per_block: usize,
    bytes: &[u8],
) -> Result<(), MetalError> {
    let rows = height.div_ceil(block.1) as usize;
    let bytes_per_row = width.div_ceil(block.0) as usize * bytes_per_block;
    let needed = rows * bytes_per_row;
    if bytes.len() < needed {
        return Err(MetalError::Unsupported(format!(
            "texture level {level} is {} bytes, needs {needed} for {width}x{height}",
            bytes.len()
        )));
    }
    unsafe {
        let _: () = msg4(
            tex.id(),
            sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
            MTLRegion::image_2d(width, height),
            level as NSUInteger,
            bytes.as_ptr() as *const c_void,
            bytes_per_row as NSUInteger,
        );
    }
    Ok(())
}

/// Fill levels 1..N on the GPU. Best effort: a device that declines the blit
/// leaves the texture with only level 0, which samples fine, just without
/// minification filtering.
fn generate_mipmaps(device: &MetalDevice, tex: &Owned) {
    let _pool = AutoreleasePool::new();
    unsafe {
        let cmd = device.command_buffer();
        if cmd.is_null() {
            return;
        }
        let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
        if blit.is_null() {
            return;
        }
        let _: () = msg1(blit, sel!("generateMipmapsForTexture:"), tex.id());
        let _: () = msg0(blit, sel!("endEncoding"));
        let _: () = msg0(cmd, sel!("commit"));
        let _: () = msg0(cmd, sel!("waitUntilCompleted"));
    }
}

/// `Texture::flip_y`, applied to the source bytes. Borrows when there is
/// nothing to do — the common case for loaders that already flipped.
fn flip_rows_if_needed(src: &Texture) -> std::borrow::Cow<'_, [u8]> {
    if !src.flip_y || src.height <= 1 {
        return std::borrow::Cow::Borrowed(&src.data[..]);
    }
    let stride = src.width as usize * src.format.bytes_per_block();
    let rows = src.height as usize;
    if src.data.len() < stride * rows {
        return std::borrow::Cow::Borrowed(&src.data[..]);
    }
    let mut out = vec![0u8; stride * rows];
    for y in 0..rows {
        let dst = y * stride;
        let s = (rows - 1 - y) * stride;
        out[dst..dst + stride].copy_from_slice(&src.data[s..s + stride]);
    }
    std::borrow::Cow::Owned(out)
}

/// Sampler parameters taken from a threers texture.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerKey {
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mipmapped: bool,
    pub wrap_s: u8,
    pub wrap_t: u8,
}

impl SamplerKey {
    pub fn from_texture(t: &Texture) -> Self {
        Self {
            mag_linear: t.mag_filter == TextureFilter::Linear,
            min_linear: t.min_filter == TextureFilter::Linear,
            mipmapped: !t.format.is_block_compressed() && t.width > 1 && t.height > 1,
            wrap_s: wrap_code(t.wrap_s),
            wrap_t: wrap_code(t.wrap_t),
        }
    }

    /// The sampler used when a material has no map: linear, clamped.
    pub fn default_linear() -> Self {
        Self {
            mag_linear: true,
            min_linear: true,
            mipmapped: false,
            wrap_s: 0,
            wrap_t: 0,
        }
    }
}

fn wrap_code(w: TextureWrap) -> u8 {
    match w {
        TextureWrap::ClampToEdge => 0,
        TextureWrap::Repeat => 1,
        TextureWrap::MirroredRepeat => 2,
    }
}

fn address_mode(code: u8) -> NSUInteger {
    match code {
        1 => address_mode::REPEAT,
        2 => address_mode::MIRROR_REPEAT,
        _ => address_mode::CLAMP_TO_EDGE,
    }
}

/// Build an `MTLSamplerState` for `key`.
pub fn new_sampler(device: &MetalDevice, key: SamplerKey) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLSamplerDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLSamplerDescriptor".into()))?;
        let filt = |linear: bool| {
            if linear {
                filter::LINEAR
            } else {
                filter::NEAREST
            }
        };
        let _: () = msg1(desc.id(), sel!("setMagFilter:"), filt(key.mag_linear));
        let _: () = msg1(desc.id(), sel!("setMinFilter:"), filt(key.min_linear));
        let _: () = msg1(
            desc.id(),
            sel!("setMipFilter:"),
            if key.mipmapped {
                filter::MIP_LINEAR
            } else {
                filter::MIP_NOT_MIPMAPPED
            },
        );
        let _: () = msg1(
            desc.id(),
            sel!("setSAddressMode:"),
            address_mode(key.wrap_s),
        );
        let _: () = msg1(
            desc.id(),
            sel!("setTAddressMode:"),
            address_mode(key.wrap_t),
        );
        let _: () = msg1(desc.id(), sel!("setMaxAnisotropy:"), 1usize);
        let state: Id = msg1(
            device.device_id(),
            sel!("newSamplerStateWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(state).ok_or_else(|| MetalError::Allocation("MTLSamplerState".into()))
    }
}

/// A 1x1 opaque white texture, bound wherever a material has no map so the
/// fragment shader can sample unconditionally.
pub fn white_texture(device: &MetalDevice) -> Result<Owned, MetalError> {
    let tex = new_texture_2d(
        device,
        1,
        1,
        pixel_format::RGBA8_UNORM,
        texture_usage::SHADER_READ,
        cpu_writable_storage(device),
        1,
        1,
    )?;
    upload_level(&tex, 0, 1, 1, (1, 1), 4, &[255, 255, 255, 255])?;
    Ok(tex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(w: u32, h: u32, flip: bool) -> Texture {
        let mut t = Texture::new(
            w,
            h,
            TextureFormat::Rgba8UnormSrgb,
            (0..w * h * 4).map(|i| (i % 251) as u8).collect(),
        );
        t.flip_y = flip;
        t
    }

    #[test]
    fn flip_moves_whole_rows() {
        let t = tex(2, 3, true);
        let flipped = flip_rows_if_needed(&t);
        let stride = 8;
        assert_eq!(&flipped[0..stride], &t.data[2 * stride..3 * stride]);
        assert_eq!(&flipped[2 * stride..3 * stride], &t.data[0..stride]);
    }

    #[test]
    fn flip_borrows_when_disabled() {
        let t = tex(2, 3, false);
        assert!(matches!(
            flip_rows_if_needed(&t),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn pipeline_keys_distinguish_blend_and_format() {
        let a = PipelineKey {
            point_sprites: false,
            blend: false,
            layered: false,
            topology: topology_class::UNSPECIFIED,
            color_format: pixel_format::RGBA8_UNORM_SRGB,
            depth_format: pixel_format::DEPTH32_FLOAT,
            sample_count: 1,
        };
        let b = PipelineKey { blend: true, ..a };
        assert_ne!(a, b);
        let c = PipelineKey {
            sample_count: 4,
            ..a
        };
        assert_ne!(a, c);
        let d = PipelineKey {
            layered: true,
            topology: topology_class::TRIANGLE,
            ..a
        };
        assert_ne!(a, d);
        // A layered pipeline is per topology class: points and triangles need
        // different ones, because Metal ties `point_size` to the class.
        assert_ne!(
            d,
            PipelineKey {
                topology: topology_class::POINT,
                ..d
            }
        );
        assert_eq!(topology_class::of(primitive::POINT), topology_class::POINT);
        assert_eq!(topology_class::of(primitive::LINE), topology_class::LINE);
        assert_eq!(
            topology_class::of(primitive::TRIANGLE),
            topology_class::TRIANGLE
        );
    }

    #[test]
    fn sampler_key_tracks_wrap_and_filter() {
        let mut t = tex(4, 4, false);
        t.wrap_s = TextureWrap::Repeat;
        t.min_filter = TextureFilter::Nearest;
        let k = SamplerKey::from_texture(&t);
        assert_eq!(k.wrap_s, 1);
        assert!(!k.min_linear);
        assert!(k.mag_linear);
        assert!(k.mipmapped);
    }
}
