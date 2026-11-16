//! Fullscreen passes: a pipeline over the one-triangle vertex stage, and an
//! encoder that draws it into a texture. Shared by the post chain and the
//! fluid, and kept apart from both so `metal-fluid` builds without the scene
//! renderer.

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg3, sel, AutoreleasePool, Id, Owned, NIL,
    YES,
};

/// A fullscreen pipeline with a vertex stage of your own.
///
/// `vs_fullscreen` hands out `uv` already flipped, because the passes around
/// it sample a framebuffer. A pass that is not a post-process — the fluid —
/// wants the unflipped one, and its own vertex stage is how it gets it.
pub(crate) fn fullscreen_pipeline_with(
    device: &MetalDevice,
    vertex: &str,
    fragment: &str,
    color_format: NSUInteger,
) -> Result<Owned, MetalError> {
    fullscreen_pipeline_blend(device, vertex, fragment, color_format, false)
}

/// As [`fullscreen_pipeline_with`], and with `blend` the pass composites over
/// what is already in the target with premultiplied alpha — `src + dst *
/// (1 - src.a)`, which is what `BlendState::PREMULTIPLIED_ALPHA_BLENDING` is
/// on the wgpu side.
///
/// A pass that draws a picture with holes in it needs this. Without blending
/// its zero alpha lands in the target and erases whatever was underneath —
/// which, for a fluid drawn over its own background, is the whole background.
pub(crate) fn fullscreen_pipeline_blend(
    device: &MetalDevice,
    vertex: &str,
    fragment: &str,
    color_format: NSUInteger,
    blend: bool,
) -> Result<Owned, MetalError> {
    let _pool = AutoreleasePool::new();
    let vs = device.function(vertex)?;
    let fs = device.function(fragment)?;
    unsafe {
        let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
            .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
        let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
        let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
        let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
        let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
        let _: () = msg1(color, sel!("setPixelFormat:"), color_format);
        if blend {
            let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
            let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
            let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
            let _: () = msg1(color, sel!("setSourceRGBBlendFactor:"), blend_factor::ONE);
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

pub(crate) fn encode_fullscreen(
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

pub(crate) fn render_pass_descriptor(dst: Id) -> Id {
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

/// A 2D texture: `sample_count` above one makes it multisampled.
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
