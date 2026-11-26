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
    /// the opaque copy's pipelines ([`draw_texture`](Self::draw_texture))
    copies: HashMap<NSUInteger, Owned>,
    /// the premultiplied blend's ([`blend_texture`](Self::blend_texture))
    blends: HashMap<NSUInteger, Owned>,
    /// `copies`, filtered up from a smaller texture
    scaled: HashMap<NSUInteger, Owned>,
    scaled_pass: bool,
}

impl MetalOverlay {
    /// Blend `rgba` (`width × height`, straight alpha, sRGB) over the frame in
    /// `pass`, then present its drawable if it has one.
    ///
    /// `rows` are the bands that changed since the last call, `(first, end)`
    /// each; `None` uploads everything. A new size uploads everything regardless.
    pub(crate) fn composite(
        &mut self,
        device: &MetalDevice,
        rgba: &[u8],
        width: u32,
        height: u32,
        rows: Option<&[(u32, u32)]>,
        cols: Option<(u32, u32)>,
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
        let whole = [(0, height)];
        let bands = match rows {
            Some(bands) if !fresh => bands,
            _ => &whole[..],
        };
        let tex = self.texture.as_ref().map(|(t, _, _)| t.id()).expect("just made");
        // only the columns that changed (a sidebar's, not the frame's width)
        let (c0, c1) = match cols {
            Some((a, b)) if !fresh && a < b && b <= width => (a, b),
            _ => (0, width),
        };
        for &(first, end) in bands {
            let (first, end) = (first.min(height), end.min(height));
            if end <= first {
                continue;
            }
            let stride = width as usize * 4;
            let region = MTLRegion {
                origin: MTLOrigin { x: c0 as NSUInteger, y: first as NSUInteger, z: 0 },
                size: MTLSize {
                    width: (c1 - c0) as NSUInteger,
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
                    rgba[first as usize * stride + c0 as usize * 4..].as_ptr() as *const std::ffi::c_void,
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

    /// Draw `texture` (a frame the size of the pass, on this device) over
    /// the frame in `pass`, opaque — a scene drawn and post-processed
    /// offscreen, shown in a window — then present the drawable if `pass`
    /// has one.
    pub(crate) fn draw_texture(&mut self, device: &MetalDevice, texture: Id, pass: &PassAttachments) -> Result<(), MetalError> {
        self.texture_pass(device, texture, pass, false)
    }

    /// [`draw_texture`](Self::draw_texture) from a texture of any size,
    /// filtered to fill the frame.
    pub(crate) fn draw_texture_scaled(&mut self, device: &MetalDevice, texture: Id, pass: &PassAttachments) -> Result<(), MetalError> {
        self.scaled_pass = true;
        let r = self.texture_pass(device, texture, pass, false);
        self.scaled_pass = false;
        r
    }

    /// Blend `texture` (premultiplied linear colour, the size of the pass)
    /// over the frame in `pass`, then present the drawable if it has one.
    pub(crate) fn blend_texture(&mut self, device: &MetalDevice, texture: Id, pass: &PassAttachments) -> Result<(), MetalError> {
        self.texture_pass(device, texture, pass, true)
    }

    fn texture_pass(&mut self, device: &MetalDevice, texture: Id, pass: &PassAttachments, over: bool) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        let dst = if pass.resolve.is_null() { pass.color } else { pass.resolve };
        let (pipelines, fragment) = if over {
            (&mut self.blends, "fs_overlay_premultiplied")
        } else if self.scaled_pass {
            (&mut self.scaled, "fs_overlay_scaled")
        } else {
            (&mut self.copies, "fs_overlay")
        };
        let pipeline = match pipelines.get(&pass.color_format) {
            Some(p) => p.id(),
            None => {
                let p = fullscreen_pipeline_blend(device, "vs_fullscreen", fragment, pass.color_format, over)?;
                let id = p.id();
                pipelines.insert(pass.color_format, p);
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
            let load = if over { load_action::LOAD } else { load_action::DONT_CARE };
            let _: () = msg1(color, sel!("setLoadAction:"), load);
            let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
            if enc.is_null() {
                return Err(MetalError::Unsupported("texture render encoder failed".into()));
            }
            let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
            let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), texture, 0usize);
            let _: () = msg3(enc, sel!("drawPrimitives:vertexStart:vertexCount:"), primitive::TRIANGLE, 0usize, 3usize);
            let _: () = msg0(enc, sel!("endEncoding"));
            if !pass.drawable.is_null() {
                let _: () = msg1(cmd, sel!("presentDrawable:"), pass.drawable);
            }
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }
}
