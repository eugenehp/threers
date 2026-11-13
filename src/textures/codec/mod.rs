//! CPU decoders for every block-compressed [`TextureFormat`], and the packed
//! 32-bit formats.
//!
//! The renderer uses these when a GPU cannot sample a format itself — BC on a
//! phone, ETC2 and ASTC on most desktops — and the CPU path tracer, the
//! exporters and anything else that reads pixels use them always. Each block
//! decoder follows its specification's integer arithmetic, so a decoded
//! texture is what the GPU would have shown, not an approximation of it;
//! `tests/texture_formats.rs` holds them to the hardware.
//!
//! What a format decodes to is [`decoded_format`]: 8-bit RGBA (sRGB kept) for
//! colour, one or two 8-bit channels for BC4 and BC5, half-float RGBA for
//! BC6H.

mod astc;
mod bc;
mod bc6h;
mod bc7;
mod etc2;
pub mod packed;

use super::{Compression, Texture, TextureFormat};
use std::sync::Arc;

/// The uncompressed format [`decode`] produces for `format`, or `format`
/// itself when it is not block-compressed.
pub fn decoded_format(format: TextureFormat) -> TextureFormat {
    use TextureFormat::*;
    match format {
        Bc4RUnorm => R8Unorm,
        Bc5RgUnorm => Rg8Unorm,
        Bc6hRgbUfloat | Bc6hRgbFloat => Rgba16Float,
        f if f.is_block_compressed() => {
            if f.is_srgb() {
                Rgba8UnormSrgb
            } else {
                Rgba8Unorm
            }
        }
        f => f,
    }
}

/// Decode one `width x height` image of a block-compressed format into
/// [`decoded_format`]'s bytes, tightly packed rows top to bottom. `None` for
/// uncompressed formats, or when `data` is shorter than the image needs.
pub fn decode(format: TextureFormat, data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    use TextureFormat::*;
    if !format.is_block_compressed() || data.len() < format.data_len(width, height) {
        return None;
    }
    let (w, h) = (width.max(1), height.max(1));
    let (bw, bh) = format.block_dim();
    let bytes = format.bytes_per_block();
    Some(match format {
        Bc6hRgbUfloat | Bc6hRgbFloat => {
            let signed = format == Bc6hRgbFloat;
            let halves = walk::<u16>(data, w, h, bw, bh, bytes, 4, |b, out| {
                bc6h::block(b, signed, out)
            });
            halves.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        Bc4RUnorm => walk::<u8>(data, w, h, bw, bh, bytes, 1, |b, out| {
            bc::bc4_block(b, out, 1)
        }),
        Bc5RgUnorm => walk::<u8>(data, w, h, bw, bh, bytes, 2, bc::bc5_block),
        Bc1RgbaUnormSrgb | Bc1RgbaUnorm => walk::<u8>(data, w, h, bw, bh, bytes, 4, |b, out| {
            bc::bc1_block(b, false, out)
        }),
        Bc3RgbaUnormSrgb | Bc3RgbaUnorm => walk::<u8>(data, w, h, bw, bh, bytes, 4, bc::bc3_block),
        Bc7RgbaUnormSrgb | Bc7RgbaUnorm => walk::<u8>(data, w, h, bw, bh, bytes, 4, bc7::block),
        Etc2Rgb8UnormSrgb | Etc2Rgb8Unorm => {
            walk::<u8>(data, w, h, bw, bh, bytes, 4, etc2::rgb_block)
        }
        Etc2Rgba8UnormSrgb | Etc2Rgba8Unorm => {
            walk::<u8>(data, w, h, bw, bh, bytes, 4, etc2::rgba_block)
        }
        Astc { srgb, .. } => walk::<u8>(data, w, h, bw, bh, bytes, 4, |b, out| {
            astc::block(b, bw, bh, srgb, out)
        }),
        _ => unreachable!("every block format is listed"),
    })
}

/// Decode every block of an image, each into a `bw * bh * ch` row-major
/// scratch, and copy the part inside the image to its place. Edge blocks are
/// clipped: an image need not be a whole number of blocks.
#[allow(clippy::too_many_arguments)]
fn walk<T: Copy + Default>(
    data: &[u8],
    w: u32,
    h: u32,
    bw: u32,
    bh: u32,
    bytes: usize,
    ch: usize,
    mut block: impl FnMut(&[u8], &mut [T]),
) -> Vec<T> {
    let (w, h, bw, bh) = (w as usize, h as usize, bw as usize, bh as usize);
    let (nx, ny) = (w.div_ceil(bw), h.div_ceil(bh));
    let mut out = vec![T::default(); w * h * ch];
    let mut scratch = vec![T::default(); bw * bh * ch];
    for by in 0..ny {
        for bx in 0..nx {
            let o = (by * nx + bx) * bytes;
            block(&data[o..o + bytes], &mut scratch);
            for y in 0..bh {
                let py = by * bh + y;
                if py >= h {
                    break;
                }
                let cols = bw.min(w - bx * bw);
                let src = &scratch[y * bw * ch..(y * bw + cols) * ch];
                let d = (py * w + bx * bw) * ch;
                out[d..d + cols * ch].copy_from_slice(src);
            }
        }
    }
    out
}

/// `data` (an image in `format`) as f32 texels in the decoded format's
/// channel order — decoding block formats first. The sRGB curve is not
/// undone.
pub fn texels_f32(format: TextureFormat, data: &[u8], width: u32, height: u32) -> Option<Vec<f32>> {
    if format.is_block_compressed() {
        decoded_format(format).texels_f32(&decode(format, data, width, height)?)
    } else {
        format.texels_f32(data)
    }
}

/// What a GPU can sample natively, as far as texture formats go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Support {
    /// BC1..BC7.
    pub bc: bool,
    /// ETC2 and EAC.
    pub etc2: bool,
    /// ASTC, LDR, every footprint.
    pub astc: bool,
    /// `R16Unorm` and `Rgba16Unorm`.
    pub unorm16: bool,
    /// Linear filtering of 32-bit floats.
    pub float32_filterable: bool,
}

impl Support {
    /// A device that samples everything.
    pub const ALL: Support = Support {
        bc: true,
        etc2: true,
        astc: true,
        unorm16: true,
        float32_filterable: true,
    };
}

/// The format a texture in `f` is held in on a device with `support`: its own
/// where the device can sample it through filterable texture bindings,
/// otherwise the nearest one it can.
///
/// - A compressed family the GPU lacks (BC on phones, ETC2 and ASTC on most
///   desktops) decodes on the CPU to [`decoded_format`].
/// - 16-bit unorm without native support goes to 32-bit float where that
///   filters, else to 16-bit float (11 bits of the 16).
/// - 32-bit float that cannot be filtered goes to 16-bit float.
pub fn device_format(f: TextureFormat, support: Support) -> TextureFormat {
    use TextureFormat::*;
    if let Some(family) = f.compression() {
        let native = match family {
            Compression::Bc => support.bc,
            Compression::Etc2 => support.etc2,
            Compression::Astc => support.astc,
        };
        return if native { f } else { decoded_format(f) };
    }
    let f32_filter = support.float32_filterable;
    match f {
        R16Unorm | Rgba16Unorm if !support.unorm16 => match (f, f32_filter) {
            (R16Unorm, true) => R32Float,
            (R16Unorm, false) => R16Float,
            (_, true) => Rgba32Float,
            (_, false) => Rgba16Float,
        },
        R32Float if !f32_filter => R16Float,
        Rg32Float if !f32_filter => Rg16Float,
        Rgba32Float if !f32_filter => Rgba16Float,
        f => f,
    }
}

/// One image converted between formats of the same channels: decoded first
/// when `from` is compressed, then rounded once into `to`.
pub fn convert_image(
    from: TextureFormat,
    to: TextureFormat,
    data: &[u8],
    w: u32,
    h: u32,
) -> Vec<u8> {
    if from == to {
        return data.to_vec();
    }
    if from.is_block_compressed() {
        let decoded = decode(from, data, w, h).unwrap_or_else(|| {
            panic!(
                "{from:?} texture data is {} bytes, {w}x{h} needs {}",
                data.len(),
                from.data_len(w, h)
            )
        });
        return convert_image(decoded_format(from), to, &decoded, w, h);
    }
    let texels = from.texels_f32(data).expect("an uncompressed format");
    to.pack_f32(&texels).expect("an uncompressed target")
}

/// `src` in [`device_format`], every mip level converted; borrowed when the
/// device takes it as it is.
pub fn for_device(src: &Texture, support: Support) -> std::borrow::Cow<'_, Texture> {
    let (w, h) = (src.width.max(1), src.height.max(1));
    let to = device_format(src.format, support);
    if to == src.format
        || src.external_rt_id.is_some()
        || src.data.len() < src.format.data_len(w, h)
    {
        return std::borrow::Cow::Borrowed(src);
    }
    if src.format.is_block_compressed() {
        // The rules the native upload enforces, checked before decoding could
        // quietly lift them on some GPUs and not others. A ragged base size in
        // particular stays an error: no API accepts one, and decoding it
        // instead would hide a broken file behind eight times the memory.
        assert!(
            !src.flip_y,
            "flip_y on a block-compressed texture ({:?}): rows of blocks cannot \
             be reversed -- encode it the right way up",
            src.format
        );
        let (bw, bh) = src.format.block_dim();
        assert!(
            w.is_multiple_of(bw) && h.is_multiple_of(bh),
            "{:?} texture is {w}x{h}: a block-compressed base size must be a \
             multiple of {bw}x{bh} (mip levels may be ragged, the base may not)",
            src.format
        );
    }
    let mut out = src.clone();
    out.format = to;
    out.data = Arc::new(convert_image(src.format, to, &src.data, w, h));
    let (mut lw, mut lh) = (w, h);
    out.mips = src
        .mips
        .iter()
        .map_while(|level| {
            lw = (lw / 2).max(1);
            lh = (lh / 2).max(1);
            (level.len() >= src.format.data_len(lw, lh))
                .then(|| Arc::new(convert_image(src.format, to, level, lw, lh)))
        })
        .collect();
    std::borrow::Cow::Owned(out)
}
