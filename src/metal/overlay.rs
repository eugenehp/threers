//! A 2D RGBA overlay blended over a finished frame — the Metal side of
//! [`Renderer::draw_overlay`](crate::renderer::Renderer::draw_overlay).
//!
//! The overlay is straight-alpha sRGB bytes the size of the frame (an
//! [`Overlay`](crate::overlay::Overlay) canvas, or rasterised captions). It is
//! kept in one texture per size, and only the rows that changed are copied in;
//! a fullscreen triangle blends it over the frame's resolved colour, loading
//! what is there rather than clearing it.

use std::collections::HashMap;

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg1, msg2, msg3, msg4, sel, AutoreleasePool, Id, Owned};
use super::postfx::{fullscreen_pipeline_blend, render_pass_descriptor};
use super::renderer::PassAttachments;
use super::resources::{cpu_writable_storage, new_texture_2d};

/// The overlay texture and its blend pipelines, one per target format.
#[derive(Default)]
pub(crate) struct MetalOverlay {
    texture: Option<(Owned, u32, u32)>,
    pipelines: HashMap<NSUInteger, Owned>,
}

impl MetalOverlay {
    /// Blend `rgba` (`width × height`, straight alpha, sRGB) over the frame in
    /// `pass`, then present its drawable if it has one.
    ///
    /// `rows` is the band that changed since the last call, `(first, end)`;
    /// `None` uploads everything. A new size uploads everything regardless.
    pub(crate) fn composite(
        &mut self,
        device: &MetalDevice,
        rgba: &[u8],
        width: u32,
        height: u32,
        rows: Option<(u32, u32)>,
        pass: &PassAttachments,
    ) -> Result<(), MetalError> {
        if rgba.len() < (width as usize) * (height as usize) * 4 {
            return Err(MetalError::Unsupported(format!(
                "overlay of {} bytes for {width}x{height}",
                rgba.len()
            )));
        }
        let _pool = AutoreleasePool::new();
        let fresh = !matches!(&self.texture, Some((_, w, h)) if (*w, *h) == (width, height));
        if fresh {
            let tex = new_texture_2d(
                device,
                width.max(1),
                height.max(1),
                pixel_format::RGBA8_UNORM_SRGB,
                texture_usage::SHADER_READ,
                cpu_writable_storage(device),
                1,
                1,
            )?;
            self.texture = Some((tex, width, height));
        }
        let (first, end) = if fresh { (0, height) } else { rows.unwrap_or((0, height)) };
        let (first, end) = (first.min(height), end.min(height));
        let tex = self.texture.as_ref().map(|(t, _, _)| t.id()).expect("just made");
        if end > first {
            let stride = width as usize * 4;
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: first as NSUInteger, z: 0 },
                size: MTLSize {
                    width: width as NSUInteger,
                    height: (end - first) as NSUInteger,
                    depth: 1,
                },
            };
            unsafe {
                let _: () = msg4(
                    tex,
                    sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                    region,
                    0usize,
                    rgba[first as usize * stride..].as_ptr() as *const std::ffi::c_void,
                    stride as NSUInteger,
                );
            }
        }

        // Over the resolved image: a multisampled pass resolves into `resolve`,
        // which is what is presented or read back.
        let dst = if pass.resolve.is_null() { pass.color } else { pass.resolve };
        let pipeline = match self.pipelines.get(&pass.color_format) {
            Some(p) => p.id(),
            None => {
                let p = fullscreen_pipeline_blend(device, "vs_fullscreen", "fs_overlay", pass.color_format, true)?;
                let id = p.id();
                self.pipelines.insert(pass.color_format, p);
                id
            }
        };
        unsafe {
            let cmd = device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let desc = render_pass_descriptor(dst);
            let attachments: Id = msg0(desc, sel!("colorAttachments"));
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(color, sel!("setLoadAction:"), load_action::LOAD);
            let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
            if enc.is_null() {
                return Err(MetalError::Unsupported("overlay render encoder failed".into()));
            }
            let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
            let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), tex, 0usize);
            let _: () = msg3(
                enc,
                sel!("drawPrimitives:vertexStart:vertexCount:"),
                primitive::TRIANGLE,
                0usize,
                3usize,
            );
            let _: () = msg0(enc, sel!("endEncoding"));
            if !pass.drawable.is_null() {
                let _: () = msg1(cmd, sel!("presentDrawable:"), pass.drawable);
            }
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }
}
