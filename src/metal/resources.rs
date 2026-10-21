//! Textures, samplers and pipeline states.

use std::ffi::c_void;

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg4, msg6, sel, AutoreleasePool, Bool, Id,
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
    /// Shell fur (`crate::fur`): `vs_fur`/`fs_fur`, the WGSL fur shader's port.
    pub fur: bool,
    /// Read vertices as the 24-byte `LineVertex` rather than the 48-byte
    /// `Vertex`. Set for line draws, whose geometry is uploaded slim.
    pub slim_lines: bool,
    /// Expand segments into screen-space quads with coverage falloff, instead
    /// of drawing one-pixel hardware lines.
    pub wide_lines: bool,
    /// Source-alpha blending, for transparent materials.
    pub blend: bool,
    /// Draw both eyes in one pass, into two slices of an array attachment.
    pub layered: bool,
    /// Blend the vertex through its bone matrices before the model transform.
    pub skinned: bool,
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
    /// three.js `colorWrite`: `false` masks every colour channel, leaving
    /// only the depth write (an invisible occluder).
    pub color_write: bool,
    /// The order-independent transparency accumulation: `fs_oit_accum`
    /// into two half-float attachments (weighted light, coverage; and the
    /// layers' transmittance), blended additively and multiplicatively.
    pub oit_accum: bool,
}

/// Build the pipeline state for `key`.
///
/// A blended mesh draw into an 8-bit sRGB attachment blends the *encoded*
/// values, as three.js's default `WebGLRenderer` does on its canvas: the
/// fragment function reads the destination (`fs_mesh_encoded_blend`) and
/// fixed-function blending is off. Where the device cannot read the
/// framebuffer the pipeline falls back to the hardware blend, in linear light.
pub fn render_pipeline(device: &MetalDevice, key: PipelineKey) -> Result<Owned, MetalError> {
    if key.blend && pixel_format::is_srgb(key.color_format) {
        if let Ok(p) = build_render_pipeline(device, key, true) {
            return Ok(p);
        }
    }
    build_render_pipeline(device, key, false)
}

fn build_render_pipeline(
    device: &MetalDevice,
    key: PipelineKey,
    encoded_blend: bool,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let (vs, fs) = match (key.point_sprites, key.layered) {
        (false, false) if key.wide_lines => ("vs_line_quad", "fs_line_quad"),
        // Slim lines have no layered variant: the stereo path widens every
        // vertex anyway, so it keeps the full-fat format.
        (false, false) if key.slim_lines => ("vs_line", "fs_mesh"),
        // Skinning is a triangle-mesh, single-view case only: a skinned point
        // sprite or line is not a thing any loader produces, and a variant for
        // each would be four more pipelines nobody compiles.
        (false, false) if key.skinned => ("vs_skinned", "fs_mesh"),
        (false, false) if key.fur => ("vs_fur", "fs_fur"),
        (false, false) => ("vs_mesh", "fs_mesh"),
        (false, true) => ("vs_mesh_layered", "fs_mesh"),
        (true, false) => ("vs_point", "fs_point"),
        (true, true) => ("vs_point_layered", "fs_point"),
    };
    let encoded_blend = encoded_blend && (fs == "fs_mesh" || fs == "fs_point");
    if encoded_blend && !key.blend {
        return Err(MetalError::PipelineCreation("no encoded blend without blending".into()));
    }
    let fs = match (encoded_blend, fs) {
        (true, "fs_point") => "fs_point_encoded_blend",
        (true, _) => "fs_mesh_encoded_blend",
        (false, fs) => fs,
    };
    let fs = if key.oit_accum { "fs_oit_accum" } else { fs };
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
        if key.oit_accum {
            // Accumulation: weighted light added up; transmittance multiplied
            // down (`dst · (1 − src)`), per channel and in alpha alike.
            for (index, (src, dst)) in [
                (blend_factor::ONE, blend_factor::ONE),
                (blend_factor::ZERO, blend_factor::ONE_MINUS_SOURCE_COLOR),
            ]
            .into_iter()
            .enumerate()
            {
                let a: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), index);
                let _: () = msg1(a, sel!("setPixelFormat:"), pixel_format::RGBA16_FLOAT);
                let _: () = msg1(a, sel!("setBlendingEnabled:"), YES);
                let _: () = msg1(a, sel!("setRgbBlendOperation:"), blend_op::ADD);
                let _: () = msg1(a, sel!("setAlphaBlendOperation:"), blend_op::ADD);
                let _: () = msg1(a, sel!("setSourceRGBBlendFactor:"), src);
                let _: () = msg1(a, sel!("setDestinationRGBBlendFactor:"), dst);
                let alpha_dst = if index == 0 { blend_factor::ONE } else { blend_factor::ONE_MINUS_SOURCE_ALPHA };
                let _: () = msg1(a, sel!("setSourceAlphaBlendFactor:"), src);
                let _: () = msg1(a, sel!("setDestinationAlphaBlendFactor:"), alpha_dst);
            }
        } else if key.blend && !encoded_blend {
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
        if !key.color_write {
            // `MTLColorWriteMaskNone`.
            let _: () = msg1(color, sel!("setWriteMask:"), 0 as NSUInteger);
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
    new_texture_array_levels(device, width, height, pixel_format, usage, slices, 1)
}

/// [`new_texture_array`] with `levels` mip levels.
pub fn new_texture_array_levels(
    device: &MetalDevice,
    width: u32,
    height: u32,
    pixel_format: NSUInteger,
    usage: NSUInteger,
    slices: u32,
    levels: u32,
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
        let _: () = msg1(desc.id(), sel!("setMipmapLevelCount:"), levels.max(1) as NSUInteger);
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
    upload_texture_as(device, src, false)
}

/// Upload `src`, optionally forcing its *linear* pixel format.
///
/// A Metal texture's pixel format is fixed at creation, so unlike wgpu — which
/// can hand out an sRGB and a non-sRGB view of the same bytes — a texture used
/// both as a colour map and as data has to be uploaded twice. See
/// `MetalRenderer::ensure_texture_as` for why the distinction matters: normal,
/// metallic-roughness and occlusion maps are quantities and decoding them
/// through a transfer curve corrupts them.
pub fn upload_texture_as(
    device: &MetalDevice,
    src: &Texture,
    linear: bool,
) -> Result<Owned, MetalError> {
    let bc = src.format.is_block_compressed();
    let format = match (linear, src.format) {
        (true, TextureFormat::Rgba8UnormSrgb) => TextureFormat::Rgba8Unorm,
        _ => src.format,
    };
    let pixel_format = pixel_format_for(format, supports_bc(device))?;
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

/// Overwrite level 0 of an uncompressed 2D texture from CPU memory.
///
/// The texture has to be `SHARED` or `MANAGED` — `replaceRegion` is a memcpy
/// into the resource, and a `PRIVATE` one has no CPU-visible storage to copy
/// into.
#[cfg(feature = "fluid")]
pub(crate) fn upload_texture_2d(
    tex: &Owned,
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    bytes: &[u8],
) -> Result<(), MetalError> {
    upload_level(tex, 0, width, height, (1, 1), bytes_per_pixel, bytes)
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
    /// Between mip levels: linear (three.js's `*MipmapLinearFilter`) or
    /// nearest (`*MipmapNearestFilter`). Only read when `mipmapped`.
    pub mip_linear: bool,
    pub wrap_s: u8,
    pub wrap_t: u8,
}

impl SamplerKey {
    pub fn from_texture(t: &Texture) -> Self {
        Self {
            mag_linear: t.mag_filter == TextureFilter::Linear,
            min_linear: t.min_filter == TextureFilter::Linear,
            // No mipmaps where the texture asks for none (`LinearFilter` /
            // `NearestFilter` as its `minFilter`).
            mipmapped: t.mipmap_filter.is_some()
                && !t.format.is_block_compressed()
                && t.width > 1
                && t.height > 1,
            mip_linear: t.mipmap_filter != Some(TextureFilter::Nearest),
            wrap_s: wrap_code(t.wrap_s),
            wrap_t: wrap_code(t.wrap_t),
        }
    }

    /// A cube texture's: its filters and mipmap mode, clamped.
    pub fn from_cube(c: &crate::textures::CubeTexture) -> Self {
        Self {
            mag_linear: c.mag_filter == TextureFilter::Linear,
            min_linear: c.min_filter == TextureFilter::Linear,
            mipmapped: c.mipmap_filter.is_some(),
            mip_linear: c.mipmap_filter != Some(TextureFilter::Nearest),
            wrap_s: 0,
            wrap_t: 0,
        }
    }

    /// The scene environment's: trilinear across its mip chain, clamped. A
    /// rough surface reads the environment by roughness down that chain.
    pub fn environment() -> Self {
        Self {
            mag_linear: true,
            min_linear: true,
            mipmapped: true,
            mip_linear: true,
            wrap_s: 0,
            wrap_t: 0,
        }
    }

    /// The sampler used when a material has no map: linear, clamped.
    pub fn default_linear() -> Self {
        Self {
            mag_linear: true,
            min_linear: true,
            mipmapped: false,
            mip_linear: true,
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

/// A 1x1, one-layer depth array, bound at the shadow slots when nothing
/// casts: Metal requires every resource a function declares to be bound, and
/// the shader declares the maps as `depth2d_array`.
pub fn empty_shadow_map(device: &MetalDevice) -> Result<Owned, MetalError> {
    new_texture_array_levels(
        device,
        1,
        1,
        pixel_format::DEPTH32_FLOAT,
        texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
        1,
        1,
    )
}

/// The comparison sampler the percentage-closer filter uses.
///
/// `compare_function = less_equal` is what makes each tap a depth *test* rather
/// than a fetch: the hardware does nine comparisons and the shader averages
/// nine zeros and ones, instead of nine fetches plus arithmetic. Linear
/// filtering on top of that is what turns the average into a gradient.
pub fn shadow_sampler(device: &MetalDevice) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLSamplerDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLSamplerDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setMagFilter:"), filter::LINEAR);
        let _: () = msg1(desc.id(), sel!("setMinFilter:"), filter::LINEAR);
        // Clamp: a lookup outside the light's frustum must read "lit", and
        // `shadow_factor` returns early for those anyway.
        let _: () = msg1(desc.id(), sel!("setSAddressMode:"), address_mode::CLAMP_TO_EDGE);
        let _: () = msg1(desc.id(), sel!("setTAddressMode:"), address_mode::CLAMP_TO_EDGE);
        let _: () = msg1(desc.id(), sel!("setCompareFunction:"), compare::LESS_EQUAL);
        let sampler: Id = msg1(
            device.device_id(),
            sel!("newSamplerStateWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(sampler)
            .ok_or_else(|| MetalError::Allocation("shadow sampler".into()))
    }
}

/// The depth-only pipeline that fills the shadow map.
///
/// No fragment function and no colour attachment: the map is a depth buffer,
/// and a fragment stage returning a colour nobody reads would cost an
/// attachment and the bandwidth to write it.
pub fn shadow_pipeline(device: &MetalDevice) -> Result<Owned, MetalError> {
    shadow_pipeline_named(device, "vs_shadow", None)
}

/// The point-light variant, which needs a fragment stage to write radial
/// distance rather than projected depth. See `fs_shadow_point`.
pub fn point_shadow_pipeline(device: &MetalDevice) -> Result<Owned, MetalError> {
    shadow_pipeline_named(device, "vs_shadow_point", Some("fs_shadow_point"))
}

/// The directional/spot pass for a cut-out caster: `fs_shadow_cut` discards
/// where the caster's map is under its `alphaTest`.
pub fn cut_shadow_pipeline(device: &MetalDevice) -> Result<Owned, MetalError> {
    shadow_pipeline_named(device, "vs_shadow_cut", Some("fs_shadow_cut"))
}

fn shadow_pipeline_named(
    device: &MetalDevice,
    vs: &str,
    fs: Option<&str>,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let vertex_fn = device.function(vs)?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vertex_fn.id());
        if let Some(fs) = fs {
            let fragment_fn = device.function(fs)?;
            let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fragment_fn.id());
        }
        let _: () = msg1(
            desc.id(),
            sel!("setDepthAttachmentPixelFormat:"),
            pixel_format::DEPTH32_FLOAT,
        );
        let mut err: Id = NIL;
        let pipeline: Id = msg2(
            device.device_id(),
            sel!("newRenderPipelineStateWithDescriptor:error:"),
            desc.id(),
            &mut err as *mut Id,
        );
        Owned::from_retained(pipeline).ok_or_else(|| {
            MetalError::PipelineCreation(error_message(err))
        })
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
            if key.mipmapped && key.mip_linear {
                filter::MIP_LINEAR
            } else if key.mipmapped {
                filter::MIP_NEAREST
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
/// Allocate a cube texture with a full mip chain.
/// A 1×1 single-slice RGBA16F array of transparent black: the transmission
/// capture's stand-in, bound when nothing in the frame is transmissive.
pub fn black_texture_array(device: &MetalDevice) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let tex = unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setTextureType:"), texture_type::TYPE_2D_ARRAY);
        let _: () = msg1(desc.id(), sel!("setPixelFormat:"), pixel_format::RGBA16_FLOAT);
        let _: () = msg1(desc.id(), sel!("setWidth:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setHeight:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setDepth:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setArrayLength:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setMipmapLevelCount:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setUsage:"), texture_usage::SHADER_READ);
        let _: () = msg1(desc.id(), sel!("setStorageMode:"), cpu_writable_storage(device));
        let tex: Id = msg1(device.device_id(), sel!("newTextureWithDescriptor:"), desc.id());
        Owned::from_retained(tex).ok_or_else(|| MetalError::Allocation("capture stand-in".into()))?
    };
    let zero = [0u16; 4];
    unsafe {
        let _: () = msg6(
            tex.id(),
            sel!("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
            MTLRegion::image_2d(1, 1),
            0usize,
            0usize,
            zero.as_ptr() as *const c_void,
            8 as NSUInteger,
            0usize,
        );
    }
    Ok(tex)
}

/// three.js's `RectAreaLight` tables (`ltc_1`, `ltc_2`, extracted from
/// three.js r165 — see `upload_ltc` in the wgpu renderer) as a two-slice
/// 64×64 half-float array, which every Apple GPU filters.
pub fn upload_ltc(device: &MetalDevice) -> Result<Owned, MetalError> {
    const LTC_1: &[u8] = include_bytes!("../renderer/ltc/ltc_float_1.bin");
    const LTC_2: &[u8] = include_bytes!("../renderer/ltc/ltc_float_2.bin");
    let _pool = AutoreleasePool::new();
    let tex = unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setTextureType:"), texture_type::TYPE_2D_ARRAY);
        let _: () = msg1(desc.id(), sel!("setPixelFormat:"), pixel_format::RGBA16_FLOAT);
        let _: () = msg1(desc.id(), sel!("setWidth:"), 64usize);
        let _: () = msg1(desc.id(), sel!("setHeight:"), 64usize);
        let _: () = msg1(desc.id(), sel!("setDepth:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setArrayLength:"), 2usize);
        let _: () = msg1(desc.id(), sel!("setMipmapLevelCount:"), 1usize);
        let _: () = msg1(desc.id(), sel!("setUsage:"), texture_usage::SHADER_READ);
        let _: () = msg1(desc.id(), sel!("setStorageMode:"), cpu_writable_storage(device));
        let tex: Id = msg1(device.device_id(), sel!("newTextureWithDescriptor:"), desc.id());
        Owned::from_retained(tex).ok_or_else(|| MetalError::Allocation("LTC table texture".into()))?
    };
    for (slice, table) in [LTC_1, LTC_2].into_iter().enumerate() {
        let half: Vec<u16> = table
            .chunks_exact(4)
            .map(|c| crate::renderer::gpu_texture::f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        unsafe {
            let _: () = msg6(
                tex.id(),
                sel!("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
                MTLRegion::image_2d(64, 64),
                0usize,
                slice as NSUInteger,
                half.as_ptr() as *const c_void,
                (64 * 8) as NSUInteger,
                0usize,
            );
        }
    }
    Ok(tex)
}

fn new_texture_cube(
    device: &MetalDevice,
    size: u32,
    pixel_format: NSUInteger,
    mip_levels: u32,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLTextureDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLTextureDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setTextureType:"), texture_type::TYPE_CUBE);
        let _: () = msg1(desc.id(), sel!("setPixelFormat:"), pixel_format);
        let _: () = msg1(desc.id(), sel!("setWidth:"), size.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setHeight:"), size.max(1) as NSUInteger);
        let _: () = msg1(desc.id(), sel!("setDepth:"), 1usize);
        let _: () = msg1(
            desc.id(),
            sel!("setMipmapLevelCount:"),
            mip_levels.max(1) as NSUInteger,
        );
        let _: () = msg1(desc.id(), sel!("setUsage:"), texture_usage::SHADER_READ);
        let _: () = msg1(desc.id(), sel!("setStorageMode:"), cpu_writable_storage(device));
        let tex: Id = msg1(
            device.device_id(),
            sel!("newTextureWithDescriptor:"),
            desc.id(),
        );
        Owned::from_retained(tex)
            .ok_or_else(|| MetalError::Allocation(format!("{size}x{size} cube texture")))
    }
}

/// Write one mip level of one cube face.
fn upload_cube_level(
    tex: &Owned,
    face: u32,
    level: u32,
    size: u32,
    bytes: &[u8],
) -> Result<(), MetalError> {
    let bytes_per_row = size as usize * 4;
    let needed = bytes_per_row * size as usize;
    if bytes.len() < needed {
        return Err(MetalError::Unsupported(format!(
            "cube face {face} level {level} is {} bytes, needs {needed}",
            bytes.len()
        )));
    }
    unsafe {
        let _: () = msg6(
            tex.id(),
            sel!("replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"),
            MTLRegion::image_2d(size, size),
            level as NSUInteger,
            face as NSUInteger,
            bytes.as_ptr() as *const c_void,
            bytes_per_row as NSUInteger,
            0usize,
        );
    }
    Ok(())
}

/// Upload a PMREM's CubeUV atlas as the 2D texture `textureCubeUV` reads:
/// half floats from its linear pixels where it has them (values above 1
/// survive), else its 8-bit pixels in the cube's own format. Rows top first,
/// as the wgpu path uploads it — the lookup flips v to match.
pub fn upload_cube_uv_atlas(
    device: &MetalDevice,
    atlas: &crate::textures::CubeUvAtlas,
    srgb: bool,
) -> Result<Owned, MetalError> {
    let usage = texture_usage::SHADER_READ;
    let storage = cpu_writable_storage(device);
    match &atlas.pixels_f32 {
        Some(f) => {
            let tex = new_texture_2d(device, atlas.width, atlas.height, pixel_format::RGBA16_FLOAT, usage, storage, 1, 1)?;
            let halves: Vec<u8> = f
                .iter()
                .flat_map(|v| crate::renderer::gpu_texture::f32_to_f16_bits(*v).to_le_bytes())
                .collect();
            upload_level(&tex, 0, atlas.width, atlas.height, (1, 1), 8, &halves)?;
            Ok(tex)
        }
        None => {
            let format = if srgb { pixel_format::RGBA8_UNORM_SRGB } else { pixel_format::RGBA8_UNORM };
            let tex = new_texture_2d(device, atlas.width, atlas.height, format, usage, storage, 1, 1)?;
            upload_level(&tex, 0, atlas.width, atlas.height, (1, 1), 4, &atlas.pixels)?;
            Ok(tex)
        }
    }
}

/// Upload a [`CubeTexture`](crate::textures::CubeTexture) as a mipmapped Metal cube.
///
/// The chain comes from the same `mip_chain` the wgpu path uses, which is the
/// point: a box downsample in *linear* light, because sRGB is a transfer
/// function and not a quantity you can take the mean of, and an environment is
/// mostly edges. Two renderers that filter the environment differently
/// disagree about what a rough surface reflects, and that is exactly the kind
/// of difference nobody can see is a bug.
pub fn upload_cube(
    device: &MetalDevice,
    cube: &crate::textures::CubeTexture,
) -> Result<Owned, MetalError> {
    let (levels, sizes) = crate::renderer::gpu_texture::mip_chain(cube.format, cube.size, &cube.faces);
    let format = match cube.format {
        TextureFormat::Rgba8UnormSrgb => pixel_format::RGBA8_UNORM_SRGB,
        _ => pixel_format::RGBA8_UNORM,
    };
    let tex = new_texture_cube(device, cube.size, format, levels.len() as u32)?;
    for (level, (faces, size)) in levels.iter().zip(&sizes).enumerate() {
        for (face, data) in faces.iter().enumerate() {
            upload_cube_level(&tex, face as u32, level as u32, *size, data)?;
        }
    }
    Ok(tex)
}

/// A 1×1 black cube, bound when a scene has no environment.
///
/// Metal requires every resource a shader declares to be bound, so the choice
/// is between this and a second pipeline variant for the no-environment case.
/// Black rather than white because if the "is there an environment" flag were
/// ever wrong, black adds nothing and white would add a uniform glow that looks
/// like a plausible lighting decision.
pub fn black_cube(device: &MetalDevice) -> Result<Owned, MetalError> {
    let tex = new_texture_cube(device, 1, pixel_format::RGBA8_UNORM, 1)?;
    for face in 0..6 {
        upload_cube_level(&tex, face, 0, 1, &[0, 0, 0, 255])?;
    }
    Ok(tex)
}

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

/// The order-independent transparency composite, in two halves over what is
/// behind (`fs_oit_tint`: times the layers' transmittance; `fs_oit_light`:
/// plus their light), into `color_format` at `sample_count`.
pub fn oit_composite_pipeline(
    device: &MetalDevice,
    color_format: NSUInteger,
    sample_count: u32,
    light: bool,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let vertex_fn = device.function("vs_oit_fullscreen")?;
    let fragment_fn = device.function(if light { "fs_oit_light" } else { "fs_oit_tint" })?;
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vertex_fn.id());
        let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fragment_fn.id());
        let _: () = msg1(desc.id(), sel!("setRasterSampleCount:"), sample_count.max(1) as NSUInteger);
        let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setPixelFormat:"), color_format);
        let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
        let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
        let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
        let (rgb_src, rgb_dst, a_src, a_dst) = if light {
            (blend_factor::ONE, blend_factor::ONE, blend_factor::ONE, blend_factor::ONE_MINUS_SOURCE_ALPHA)
        } else {
            (blend_factor::ZERO, blend_factor::SOURCE_COLOR, blend_factor::ZERO, blend_factor::ONE)
        };
        let _: () = msg1(color, sel!("setSourceRGBBlendFactor:"), rgb_src);
        let _: () = msg1(color, sel!("setDestinationRGBBlendFactor:"), rgb_dst);
        let _: () = msg1(color, sel!("setSourceAlphaBlendFactor:"), a_src);
        let _: () = msg1(color, sel!("setDestinationAlphaBlendFactor:"), a_dst);
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
            oit_accum: false,
            fur: false,
            point_sprites: false,
            slim_lines: false,
            wide_lines: false,
            blend: false,
            layered: false,
            skinned: false,
            topology: topology_class::UNSPECIFIED,
            color_format: pixel_format::RGBA8_UNORM_SRGB,
            depth_format: pixel_format::DEPTH32_FLOAT,
            sample_count: 1,
            color_write: true,
        };
        let b = PipelineKey { blend: true, ..a };
        assert_ne!(a, b);
        assert_ne!(a, PipelineKey { color_write: false, ..a });
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
        // Slim lines read a different vertex struct, so they cannot share a
        // pipeline with the mesh path.
        assert_ne!(a, PipelineKey { slim_lines: true, ..a });
        assert_ne!(a, PipelineKey { wide_lines: true, ..a });
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

