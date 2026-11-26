//! Pictures drawn by the GPU under a UI's overlay: each an RGBA8 image kept
//! as a texture (uploaded once, by the image's identity), drawn into a
//! rectangle of the frame, cut to a rectangle, filtered as it is scaled.
//! [`MetalUi`](super::MetalUi) draws a UI's opaque images so, leaving a
//! hole in its overlay where each goes: a scrolled or zoomed page of them
//! moves on the GPU, not by painting its pixels again.

use std::collections::HashMap;

use super::device::{compile_library, MetalDevice, MetalError};
use super::enums::*;
use super::objc::{alloc_init, class, error_message, msg0, msg1, msg2, msg3, msg4, sel, AutoreleasePool, Id, Owned, NIL, YES};
use super::pass::new_texture_2d;
use super::renderer::PassAttachments;
use super::resources::{cpu_writable_storage, library_function};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct QuadParams {
    float4 rect;   // left, top, right, bottom in clip space
    float opacity;
    float3 pad;
};

struct QuadOut {
    float4 position [[position]];
    float2 uv;
};

vertex QuadOut vs_ui_quad(uint vid [[vertex_id]], constant QuadParams& p [[buffer(0)]]) {
    const float2 c = float2(float(vid & 1u), float((vid >> 1u) & 1u));
    QuadOut out;
    out.position = float4(mix(p.rect.xy, p.rect.zw, c), 0.0, 1.0);
    out.uv = c;
    return out;
}

fragment float4 fs_ui_quad(
    QuadOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant QuadParams& p [[buffer(0)]])
{
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    const float4 c = src.sample(s, in.uv);
    const float a = c.a * p.opacity;
    return float4(c.rgb * a, a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct QuadParams {
    rect: [f32; 4],
    opacity: f32,
    pad: [f32; 3],
}

/// One picture to draw: where (frame pixels, `x, y, width, height`), cut
/// to `clip` (whole pixels), how opaque.
#[derive(Debug, Clone)]
pub(crate) struct Quad {
    pub image: threers_ui::image::ImageRef,
    pub dest: [f32; 4],
    pub clip: [u32; 4],
    pub opacity: f32,
}

/// The textures, by image, and the pipelines that draw them.
#[derive(Default)]
pub(crate) struct MetalQuads {
    library: Option<Owned>,
    pipelines: HashMap<NSUInteger, Owned>,
    /// each image's texture, and the frame it was last drawn in
    textures: HashMap<u64, (Owned, u64)>,
    frame: u64,
}

/// Frames a texture is kept undrawn (a page scrolled back in soon after).
const KEEP: u64 = 90;

impl MetalQuads {
    /// Draws `quads` over the frame in `pass`, in order (it is not
    /// presented: the overlay goes over them).
    pub(crate) fn draw(&mut self, device: &MetalDevice, quads: &[Quad], pass: &PassAttachments) -> Result<(), MetalError> {
        self.frame += 1;
        if quads.is_empty() {
            self.evict();
            return Ok(());
        }
        let _pool = AutoreleasePool::new();
        let pipeline = self.pipeline(device, pass.color_format)?;
        let mut textures = Vec::with_capacity(quads.len());
        for q in quads {
            textures.push(self.texture(device, &q.image)?);
        }
        let dst = if pass.resolve.is_null() { pass.color } else { pass.resolve };
        let (fw, fh) = (pass.width.max(1) as f32, pass.height.max(1) as f32);
        unsafe {
            let cmd = device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let desc = super::pass::render_pass_descriptor(dst);
            let attachments: Id = msg0(desc, sel!("colorAttachments"));
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(color, sel!("setLoadAction:"), load_action::LOAD);
            let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
            if enc.is_null() {
                return Err(MetalError::Unsupported("image quads render encoder failed".into()));
            }
            let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
            for (q, tex) in quads.iter().zip(textures) {
                let [cx, cy, cw, ch] = q.clip;
                let (cx, cy) = (cx.min(pass.width), cy.min(pass.height));
                let (cw, ch) = (cw.min(pass.width - cx), ch.min(pass.height - cy));
                if cw == 0 || ch == 0 {
                    continue;
                }
                let scissor = MTLScissorRect { x: cx as NSUInteger, y: cy as NSUInteger, width: cw as NSUInteger, height: ch as NSUInteger };
                let _: () = msg1(enc, sel!("setScissorRect:"), scissor);
                let [x, y, w, h] = q.dest;
                let params = QuadParams {
                    rect: [x / fw * 2.0 - 1.0, 1.0 - y / fh * 2.0, (x + w) / fw * 2.0 - 1.0, 1.0 - (y + h) / fh * 2.0],
                    opacity: q.opacity,
                    pad: [0.0; 3],
                };
                let bytes = &params as *const QuadParams as *const std::ffi::c_void;
                let len = std::mem::size_of::<QuadParams>();
                let _: () = msg3(enc, sel!("setVertexBytes:length:atIndex:"), bytes, len, 0usize);
                let _: () = msg3(enc, sel!("setFragmentBytes:length:atIndex:"), bytes, len, 0usize);
                let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), tex, 0usize);
                let _: () = msg3(enc, sel!("drawPrimitives:vertexStart:vertexCount:"), primitive::TRIANGLE_STRIP, 0usize, 4usize);
            }
            let _: () = msg0(enc, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        self.evict();
        Ok(())
    }

    /// Textures not drawn for a while let go.
    fn evict(&mut self) {
        let now = self.frame;
        self.textures.retain(|_, (_, last)| now - *last <= KEEP);
    }

    /// `image`'s texture: made and filled the first time it is drawn.
    fn texture(&mut self, device: &MetalDevice, image: &threers_ui::image::ImageRef) -> Result<Id, MetalError> {
        let frame = self.frame;
        if let Some((t, last)) = self.textures.get_mut(&image.id()) {
            *last = frame;
            return Ok(t.id());
        }
        let (w, h) = (image.width.max(1), image.height.max(1));
        let tex = new_texture_2d(device, w, h, pixel_format::RGBA8_UNORM_SRGB, texture_usage::SHADER_READ, cpu_writable_storage(device), 1, 1)?;
        if image.rgba.len() >= w as usize * h as usize * 4 {
            let region = MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w as NSUInteger, height: h as NSUInteger, depth: 1 } };
            unsafe {
                let _: () = msg4(
                    tex.id(),
                    sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                    region,
                    0usize,
                    image.rgba.as_ptr() as *const std::ffi::c_void,
                    (w as usize * 4) as NSUInteger,
                );
            }
        }
        let id = tex.id();
        self.textures.insert(image.id(), (tex, frame));
        Ok(id)
    }

    fn pipeline(&mut self, device: &MetalDevice, format: NSUInteger) -> Result<Id, MetalError> {
        if let Some(p) = self.pipelines.get(&format) {
            return Ok(p.id());
        }
        if self.library.is_none() {
            self.library = Some(unsafe { compile_library(device.device_id(), SOURCE)? });
        }
        let library = self.library.as_ref().map(|l| l.id()).expect("just compiled");
        let vs = library_function(library, "vs_ui_quad")?;
        let fs = library_function(library, "fs_ui_quad")?;
        let state = unsafe {
            let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
                .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
            let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
            let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
            let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(color, sel!("setPixelFormat:"), format);
            // premultiplied over what is there
            let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
            let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
            let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
            let _: () = msg1(color, sel!("setSourceRGBBlendFactor:"), blend_factor::ONE);
            let _: () = msg1(color, sel!("setDestinationRGBBlendFactor:"), blend_factor::ONE_MINUS_SOURCE_ALPHA);
            let _: () = msg1(color, sel!("setSourceAlphaBlendFactor:"), blend_factor::ONE);
            let _: () = msg1(color, sel!("setDestinationAlphaBlendFactor:"), blend_factor::ONE_MINUS_SOURCE_ALPHA);
            let mut err: Id = NIL;
            let state: Id = msg2(device.device_id(), sel!("newRenderPipelineStateWithDescriptor:error:"), desc.id(), &mut err as *mut Id);
            Owned::from_retained(state).ok_or_else(|| MetalError::PipelineCreation(error_message(err)))?
        };
        let id = state.id();
        self.pipelines.insert(format, state);
        Ok(id)
    }
}
