//! Offscreen colour + depth attachments, and reading them back.

use super::device::{MetalDevice, MetalError};
use super::enums::*;
use super::objc::{msg0, msg9, sel, AutoreleasePool, Id, Owned, NIL};
use super::renderer::PassAttachments;
use super::resources::{new_texture_2d, new_texture_array};

/// Rows of a texture→buffer copy are padded to this many bytes.
///
/// Metal's own requirement is smaller and format-dependent (and queryable per
/// device); 256 is a multiple of every value it can take, costs a few KB on a
/// 4K frame, and removes the question.
const ROW_ALIGNMENT: usize = 256;

/// An offscreen render target: a colour texture, a depth texture, and — when
/// MSAA is on — the multisample texture the pass actually draws into plus the
/// single-sample texture it resolves to.
pub struct MetalRenderTarget {
    /// What the pass draws into: the MSAA texture, or the colour texture itself.
    draw_color: Owned,
    /// The readable single-sample colour. Same object as `draw_color` when
    /// MSAA is off.
    color: Owned,
    depth: Owned,
    width: u32,
    height: u32,
    color_format: NSUInteger,
    sample_count: u32,
    /// Array slices in the attachments — 2 for a stereo (layered) target.
    slices: u32,
}

impl MetalRenderTarget {
    /// Allocate a target of `width` x `height`.
    ///
    /// `sample_count` of 1 disables MSAA; 4 is the usual choice and is
    /// supported everywhere Metal is. `color_format` is normally
    /// [`pixel_format::RGBA8_UNORM_SRGB`] — the same choice the wgpu headless
    /// renderer makes, so the two produce comparable images.
    pub fn new(
        device: &MetalDevice,
        width: u32,
        height: u32,
        color_format: NSUInteger,
        sample_count: u32,
    ) -> Result<Self, MetalError> {
        let (width, height) = (width.max(1), height.max(1));
        let samples = sample_count.max(1);
        if samples > 1 && !device.supports_sample_count(samples) {
            return Err(MetalError::Unsupported(format!(
                "{samples}x MSAA — this device does not support that sample count"
            )));
        }

        // The resolved colour is a blit source for readback and a sampling
        // source for anything downstream, so it needs SHADER_READ as well.
        let color = new_texture_2d(
            device,
            width,
            height,
            color_format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            1,
            1,
        )?;
        let draw_color = if samples > 1 {
            new_texture_2d(
                device,
                width,
                height,
                color_format,
                texture_usage::RENDER_TARGET,
                storage_mode::PRIVATE,
                samples,
                1,
            )?
        } else {
            color.clone()
        };
        let depth = new_texture_2d(
            device,
            width,
            height,
            pixel_format::DEPTH32_FLOAT,
            // SHADER_READ as well as RENDER_TARGET: SSAO reads this depth back
            // as a `depth2d` in a fragment pass. Without the usage bit the
            // texture binds but reads as zero, which is occlusion everywhere.
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            storage_mode::PRIVATE,
            samples,
            1,
        )?;

        Ok(Self {
            draw_color,
            color,
            depth,
            width,
            height,
            color_format,
            sample_count: samples,
            slices: 1,
        })
    }

    /// Attachments for [`MetalRenderer::render`](super::MetalRenderer::render).
    pub fn attachments(&self) -> PassAttachments {
        let resolve = if self.sample_count > 1 {
            self.color.id()
        } else {
            NIL
        };
        unsafe {
            PassAttachments::from_raw(
                self.draw_color.id(),
                resolve,
                self.depth.id(),
                NIL,
                self.width,
                self.height,
                self.color_format,
                pixel_format::DEPTH32_FLOAT,
                self.sample_count,
            )
        }
    }

    /// Size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// MSAA sample count (1 = off).
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// The resolved colour texture. Borrowed — do not release.
    pub fn color_texture(&self) -> Id {
        self.color.id()
    }

    /// The depth texture. Borrowed — do not release.
    pub fn depth_texture(&self) -> Id {
        self.depth.id()
    }

    /// A target whose colour and depth attachments are texture *arrays* of
    /// `slices` slices — what stereo rendering draws into, one slice per eye.
    ///
    /// This is the visionOS `layered` layout, reproduced offscreen: the same
    /// attachments the compositor hands over, so the stereo path can be
    /// exercised (and read back) on a Mac.
    pub fn layered(
        device: &MetalDevice,
        width: u32,
        height: u32,
        color_format: NSUInteger,
        slices: u32,
    ) -> Result<Self, MetalError> {
        let (width, height) = (width.max(1), height.max(1));
        let slices = slices.clamp(1, 16);
        let color = new_texture_array(
            device,
            width,
            height,
            color_format,
            texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
            slices,
        )?;
        let depth = new_texture_array(
            device,
            width,
            height,
            pixel_format::DEPTH32_FLOAT,
            texture_usage::RENDER_TARGET,
            slices,
        )?;
        Ok(Self {
            draw_color: color.clone(),
            color,
            depth,
            width,
            height,
            color_format,
            sample_count: 1,
            slices,
        })
    }

    /// Array slices in the attachments. 1 unless this is a
    /// [`layered`](Self::layered) target.
    pub fn slices(&self) -> u32 {
        self.slices
    }

    /// Read the colour attachment back as tightly-packed 8-bit RGBA, top row
    /// first — the same layout [`crate::encode_png`] expects.
    ///
    /// Blocks until the GPU has finished: the copy is committed on the same
    /// queue as the draw, so it cannot observe a half-drawn frame.
    pub fn read_rgba(&self, device: &MetalDevice) -> Result<Vec<u8>, MetalError> {
        let mut out = Vec::new();
        self.read_rgba_into(device, &mut out)?;
        Ok(out)
    }

    /// Unpack the colour attachment into `out`, reusing its capacity when the
    /// size matches. Avoids a fresh allocation every frame in video export.
    pub fn read_rgba_into(
        &self,
        device: &MetalDevice,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        self.read_rgba_slice_into(device, 0, out)
    }

    /// Read one array slice back — eye `slice` of a
    /// [`layered`](Self::layered) target.
    pub fn read_rgba_slice(
        &self,
        device: &MetalDevice,
        slice: u32,
    ) -> Result<Vec<u8>, MetalError> {
        let mut out = Vec::new();
        self.read_rgba_slice_into(device, slice, &mut out)?;
        Ok(out)
    }

    /// Like [`read_rgba_slice`](Self::read_rgba_slice) but reuses `out`.
    pub fn read_rgba_slice_into(
        &self,
        device: &MetalDevice,
        slice: u32,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        let (width, height, padded, bpp) = self.readback_layout()?;
        let staging = device.new_buffer(&vec![0u8; padded * height])?;
        self.blit_color_to_buffer(device, slice, &staging, padded)?;
        self.unpack_staging(&staging, width, height, padded, bpp, out)
    }

    /// Queue a texture→buffer copy without blocking. Pair with
    /// [`finish_readback_into`](Self::finish_readback_into) on the next frame.
    pub fn queue_readback(
        &self,
        device: &MetalDevice,
        staging: &Owned,
    ) -> Result<Owned, MetalError> {
        self.queue_readback_slice(device, staging, 0)
    }

    /// Queue readback for one array slice; returns the command buffer to wait on later.
    pub fn queue_readback_slice(
        &self,
        device: &MetalDevice,
        staging: &Owned,
        slice: u32,
    ) -> Result<Owned, MetalError> {
        let (_, height, padded, _) = self.readback_layout()?;
        self.blit_color_to_buffer_async(device, slice, staging, padded, height)
    }

    /// Wait for a prior [`queue_readback`](Self::queue_readback) and unpack into `out`.
    // `Id` is `*mut Object`, so clippy asks for `unsafe fn`. Every `Id` in this
    // module comes from the Metal runtime and is only ever handed back to it;
    // making the encoders unsafe would put the keyword on the whole backend
    // without making any caller check anything it is not already checking.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn finish_readback_into(
        &self,
        cmd: Id,
        staging: &Owned,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        unsafe {
            if !cmd.is_null() {
                let _: () = msg0(cmd, sel!("waitUntilCompleted"));
            }
        }
        let (width, height, padded, bpp) = self.readback_layout()?;
        self.unpack_staging(staging, width, height, padded, bpp, out)
    }

    fn readback_layout(&self) -> Result<(usize, usize, usize, usize), MetalError> {
        let bpp = pixel_format::bytes_per_pixel(self.color_format).ok_or_else(|| {
            MetalError::Unsupported(format!(
                "readback of pixel format {} is not implemented",
                self.color_format
            ))
        })?;
        if bpp != 4 {
            return Err(MetalError::Unsupported(format!(
                "readback expects an 8-bit 4-channel format, got {} bytes per pixel",
                bpp
            )));
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let padded = padded_row_bytes(width, bpp);
        Ok((width, height, padded, bpp))
    }

    fn blit_color_to_buffer(
        &self,
        device: &MetalDevice,
        slice: u32,
        staging: &Owned,
        padded: usize,
    ) -> Result<(), MetalError> {
        let cmd =
            self.blit_color_to_buffer_async(device, slice, staging, padded, self.height as usize)?;
        let _pool = AutoreleasePool::new();
        unsafe {
            let _: () = msg0(cmd.id(), sel!("waitUntilCompleted"));
        }
        Ok(())
    }

    fn blit_color_to_buffer_async(
        &self,
        device: &MetalDevice,
        slice: u32,
        staging: &Owned,
        padded: usize,
        height: usize,
    ) -> Result<Owned, MetalError> {
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
            if blit.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let _: () = msg9(
                blit,
                sel!(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"
                ),
                self.color.id(),
                slice.min(self.slices.saturating_sub(1)) as NSUInteger,
                0usize,
                MTLOrigin::default(),
                MTLSize {
                    width: self.width as NSUInteger,
                    height: self.height as NSUInteger,
                    depth: 1,
                },
                staging.id(),
                0usize,
                padded as NSUInteger,
                (padded * height) as NSUInteger,
            );
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
            // `[queue commandBuffer]` is autoreleased, and the pool above pops
            // when this returns. Handing the raw pointer back leaves the caller
            // messaging an object whose only remaining owner is the queue —
            // retain it here, inside the pool that owns it, so waiting on it
            // later is defined.
            Owned::retain(cmd).ok_or(MetalError::NoCommandQueue)
        }
    }

    fn unpack_staging(
        &self,
        staging: &Owned,
        width: usize,
        height: usize,
        padded: usize,
        bpp: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), MetalError> {
        unpack_staging_buffer(staging, width, height, padded, bpp, self.color_format, out)
    }
}

fn unpack_staging_buffer(
    staging: &Owned,
    width: usize,
    height: usize,
    padded: usize,
    bpp: usize,
    color_format: NSUInteger,
    out: &mut Vec<u8>,
) -> Result<(), MetalError> {
    let need = width * height * bpp;
    out.clear();
    out.resize(need, 0);
    let _pool = AutoreleasePool::new();
    unsafe {
        use std::ffi::c_void;
        let contents: *const u8 =
            msg0::<*mut c_void>(staging.id(), sel!("contents")).cast();
        if contents.is_null() {
            return Err(MetalError::Allocation("readback buffer contents".into()));
        }
        for y in 0..height {
            let src = std::slice::from_raw_parts(contents.add(y * padded), width * bpp);
            out[y * width * bpp..(y + 1) * width * bpp].copy_from_slice(src);
        }
    }
    if pixel_format::is_bgra(color_format) {
        for px in out.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    Ok(())
}

/// Append a texture→staging blit onto an existing (uncommitted) command buffer.
pub fn blit_texture_to_buffer_on_cmd(
    cmd: Id,
    texture: Id,
    width: u32,
    height: u32,
    staging: &Owned,
    padded: usize,
) -> Result<(), MetalError> {
    unsafe {
        let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
        if blit.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        let _: () = msg9(
            blit,
            sel!(
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"
            ),
            texture,
            0usize,
            0usize,
            MTLOrigin::default(),
            MTLSize {
                width: width as NSUInteger,
                height: height as NSUInteger,
                depth: 1,
            },
            staging.id(),
            0usize,
            padded as NSUInteger,
            (padded * height as usize) as NSUInteger,
        );
        let _: () = msg0(blit, sel!("endEncoding"));
    }
    Ok(())
}


/// Unpack a prior async texture readback (after `waitUntilCompleted` on its cmd).
pub fn finish_texture_readback_into(
    cmd: Id,
    staging: &Owned,
    width: u32,
    height: u32,
    color_format: NSUInteger,
    out: &mut Vec<u8>,
) -> Result<(), MetalError> {
    let _pool = AutoreleasePool::new();
    unsafe {
        if !cmd.is_null() {
            let _: () = msg0(cmd, sel!("waitUntilCompleted"));
        }
    }
    let bpp = pixel_format::bytes_per_pixel(color_format).ok_or_else(|| {
        MetalError::Unsupported(format!(
            "readback of pixel format {color_format} is not implemented"
        ))
    })?;
    if bpp != 4 {
        return Err(MetalError::Unsupported(
            "readback expects an 8-bit 4-channel format".into(),
        ));
    }
    let width = width as usize;
    let height = height as usize;
    let padded = padded_row_bytes(width, bpp);
    unpack_staging_buffer(staging, width, height, padded, bpp, color_format, out)
}

impl std::fmt::Debug for MetalRenderTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRenderTarget")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("color_format", &self.color_format)
            .field("sample_count", &self.sample_count)
            .finish()
    }
}

/// Padded row stride for a texture→buffer copy.
fn padded_row_bytes(width: usize, bytes_per_pixel: usize) -> usize {
    (width * bytes_per_pixel).div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_pad_up_to_the_alignment() {
        assert_eq!(padded_row_bytes(64, 4), 256);
        assert_eq!(padded_row_bytes(65, 4), 512);
        assert_eq!(padded_row_bytes(256, 4), 1024);
        // Never smaller than the unpadded row.
        for w in [1usize, 3, 17, 100, 1920] {
            assert!(padded_row_bytes(w, 4) >= w * 4);
            assert_eq!(padded_row_bytes(w, 4) % ROW_ALIGNMENT, 0);
        }
    }
}
