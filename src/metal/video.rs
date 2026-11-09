//! Streamed textures (video) on the Metal backend: the newest frame of a
//! [`TextureStream`] put into the texture the renderer already has.
//!
//! RGBA frames are copied in. YUV frames go up as three `R8Uint` or `R16Uint`
//! planes and
//! `fs_yuv` converts them into the texture, the twin of the wgpu pass in
//! `renderer/yuv.rs`. Either way the mip chain, when the texture has one, is
//! rebuilt after on the GPU. Nothing here waits for the GPU, except before
//! rewriting planes the last conversion may still be reading.

use std::collections::HashMap;
use std::ffi::c_void;

use super::device::{wait_until_completed, MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg1, msg2, msg3, msg4, sel, AutoreleasePool, Id, Owned};
use super::postfx::{fullscreen_pipeline_blend, render_pass_descriptor};
use super::resources::{cpu_writable_storage, new_texture_2d};
use crate::renderer::yuv::YuvParams;
use crate::textures::{StreamFrame, Texture, TextureStream, YuvFrame};

/// One stream's planes, and the conversion that last read them.
struct Planes {
    sizes: [(u32, u32); 3],
    formats: [NSUInteger; 3],
    textures: [Owned; 3],
    /// The command buffer of the last conversion, retained.
    reading: Option<Owned>,
}

/// The YUV pipelines and each stream's planes. See the [module docs](self).
#[derive(Default)]
pub(crate) struct MetalStreams {
    pipelines: HashMap<NSUInteger, Owned>,
    planes: HashMap<u64, Planes>,
}

impl MetalStreams {
    /// Put `stream`'s newest frame into level 0 of `tex` (made for `src`) and
    /// rebuild its mip chain. Returns the frame count written, or `None` when
    /// the stream's size is not the texture's.
    pub(crate) fn write(
        &mut self,
        device: &MetalDevice,
        tex: &Owned,
        src: &Texture,
        stream: &TextureStream,
    ) -> Result<Option<u64>, MetalError> {
        if stream.size() != (src.width, src.height) {
            return Ok(None);
        }
        let _pool = AutoreleasePool::new();
        stream.read_frame(|frame, seq| {
            match frame {
                // Nothing written yet: opaque black, not whatever the
                // memory held.
                StreamFrame::Rgba([]) => unsafe {
                    let cmd = device.command_buffer();
                    if !cmd.is_null() {
                        // The descriptor clears to (0, 0, 0, 1).
                        let desc = render_pass_descriptor(tex.id());
                        let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
                        if !enc.is_null() {
                            let _: () = msg0(enc, sel!("endEncoding"));
                        }
                        let _: () = msg0(cmd, sel!("commit"));
                    }
                    return Ok(Some(seq));
                },
                StreamFrame::Rgba(rgba) => write_rgba(tex, src, rgba)?,
                StreamFrame::Yuv(yuv) => {
                    self.convert(device, tex, src, yuv)?;
                    return Ok(Some(seq));
                }
            }
            unsafe {
                if mip_levels(tex) > 1 {
                    let cmd = device.command_buffer();
                    if !cmd.is_null() {
                        encode_mipmaps(cmd, tex);
                        let _: () = msg0(cmd, sel!("commit"));
                    }
                }
            }
            Ok(Some(seq))
        })
    }

    fn convert(&mut self, device: &MetalDevice, tex: &Owned, src: &Texture, frame: &YuvFrame) -> Result<(), MetalError> {
        if !frame.is_valid() {
            return Ok(());
        }
        let (cw, ch) = frame.chroma_size();
        let chroma = if frame.is_monochrome() { (1, 1) } else { (cw, ch) };
        let sizes = if frame.interleaved {
            [(frame.width.max(1), frame.height.max(1)), chroma, (1, 1)]
        } else {
            [(frame.width.max(1), frame.height.max(1)), chroma, chroma]
        };
        let bytes = frame.y.bytes_per_sample();
        let (one, two) = if bytes == 1 {
            (pixel_format::R8_UINT, pixel_format::RG8_UINT)
        } else {
            (pixel_format::R16_UINT, pixel_format::RG16_UINT)
        };
        let formats = if frame.interleaved { [one, two, one] } else { [one; 3] };
        if self.planes.get(&src.id).is_none_or(|p| p.sizes != sizes || p.formats != formats) {
            let make = |(w, h): (u32, u32), format: NSUInteger| {
                new_texture_2d(
                    device,
                    w,
                    h,
                    format,
                    texture_usage::SHADER_READ,
                    cpu_writable_storage(device),
                    1,
                    1,
                )
            };
            let textures = [make(sizes[0], formats[0])?, make(sizes[1], formats[1])?, make(sizes[2], formats[2])?];
            self.planes.insert(
                src.id,
                Planes {
                    sizes,
                    formats,
                    textures,
                    reading: None,
                },
            );
        }
        let format: NSUInteger = unsafe { msg0(tex.id(), sel!("pixelFormat")) };
        let pipeline = match self.pipelines.entry(format) {
            std::collections::hash_map::Entry::Occupied(p) => p.get().id(),
            std::collections::hash_map::Entry::Vacant(slot) => slot
                .insert(fullscreen_pipeline_blend(device, "vs_fullscreen", "fs_yuv", format, false)?)
                .id(),
        };
        let planes = self.planes.get_mut(&src.id).expect("made above");
        // The planes are shared memory: the last conversion has to be done
        // with them before they are written over.
        if let Some(cmd) = planes.reading.take() {
            unsafe { wait_until_completed(cmd.id())? };
        }
        let blank = [0u8; 4];
        let data: [&[u8]; 3] = if frame.is_monochrome() {
            [frame.y.as_bytes(), &blank[..bytes as usize], &blank[..bytes as usize]]
        } else if frame.interleaved {
            [frame.y.as_bytes(), frame.u.as_bytes(), &blank[..bytes as usize]]
        } else {
            [frame.y.as_bytes(), frame.u.as_bytes(), frame.v.as_bytes()]
        };
        for (((plane, &(w, h)), samples), format) in planes.textures.iter().zip(&sizes).zip(data).zip(formats) {
            let texel = if format == two { bytes * 2 } else { bytes };
            unsafe {
                let _: () = msg4(
                    plane.id(),
                    sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
                    MTLRegion::image_2d(w, h),
                    0usize,
                    samples.as_ptr() as *const c_void,
                    (w * texel) as NSUInteger,
                );
            }
        }
        let (tw, th): (NSUInteger, NSUInteger) = unsafe { (msg0(tex.id(), sel!("width")), msg0(tex.id(), sel!("height"))) };
        let scale = frame.width.div_ceil(tw.max(1) as u32).max(frame.height.div_ceil(th.max(1) as u32));
        let params = YuvParams::new(frame, src.flip_y, scale, pixel_format::is_srgb(format), th as u32);
        unsafe {
            let cmd = device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let desc = render_pass_descriptor(tex.id());
            let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
            if enc.is_null() {
                return Err(MetalError::Unsupported("video conversion encoder failed".into()));
            }
            let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
            // Interleaved: the Cb/Cr plane at both chroma slots.
            let bound = if frame.interleaved { [0, 1, 1] } else { [0, 1, 2] };
            for (i, which) in bound.into_iter().enumerate() {
                let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), planes.textures[which].id(), i as NSUInteger);
            }
            let _: () = msg3(
                enc,
                sel!("setFragmentBytes:length:atIndex:"),
                &params as *const YuvParams as *const c_void,
                std::mem::size_of::<YuvParams>() as NSUInteger,
                0usize,
            );
            let _: () = msg3(
                enc,
                sel!("drawPrimitives:vertexStart:vertexCount:"),
                primitive::TRIANGLE,
                0usize,
                3usize,
            );
            let _: () = msg0(enc, sel!("endEncoding"));
            if mip_levels(tex) > 1 {
                encode_mipmaps(cmd, tex);
            }
            let _: () = msg0(cmd, sel!("commit"));
            planes.reading = Owned::retain(cmd);
        }
        Ok(())
    }
}

/// Copy an RGBA frame into level 0, turned over when the texture asks.
fn write_rgba(tex: &Owned, src: &Texture, rgba: &[u8]) -> Result<(), MetalError> {
    let stride = src.width as usize * 4;
    if rgba.len() < stride * src.height as usize {
        return Err(MetalError::Unsupported(format!(
            "a {}-byte frame for a {}x{} stream",
            rgba.len(),
            src.width,
            src.height
        )));
    }
    let turned;
    let bytes = if src.flip_y && src.height > 1 {
        turned = rgba.chunks_exact(stride).rev().flatten().copied().collect::<Vec<u8>>();
        &turned[..]
    } else {
        rgba
    };
    unsafe {
        let _: () = msg4(
            tex.id(),
            sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
            MTLRegion::image_2d(src.width, src.height),
            0usize,
            bytes.as_ptr() as *const c_void,
            stride as NSUInteger,
        );
    }
    Ok(())
}

unsafe fn mip_levels(tex: &Owned) -> NSUInteger {
    unsafe { msg0(tex.id(), sel!("mipmapLevelCount")) }
}

unsafe fn encode_mipmaps(cmd: Id, tex: &Owned) {
    let blit: Id = unsafe { msg0(cmd, sel!("blitCommandEncoder")) };
    if !blit.is_null() {
        let _: () = unsafe { msg1(blit, sel!("generateMipmapsForTexture:"), tex.id()) };
        let _: () = unsafe { msg0(blit, sel!("endEncoding")) };
    }
}
