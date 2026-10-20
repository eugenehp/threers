//! Getting pixels back off the device, and the two things that always go wrong.
//!
//! **Row padding.** `copy_texture_to_buffer` requires each row to start on a
//! 256-byte boundary, so the buffer that comes back is wider than the image
//! unless the width happens to divide. Forgetting that produces a picture that
//! shears progressively to one side — recognisable once you have seen it, and
//! mystifying the first time.
//!
//! **Half floats.** Every float target in this crate is `*16Float`, and
//! nothing in `std` decodes one. Eight separate copies of a `half_to_f32`
//! grew in this crate before this module existed; they are now all [`f16_to_f32`].
//!
//! ```no_run
//! use threers::renderer::readback;
//! # fn demo(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
//! let rgba = readback::texture_bytes(device, queue, texture);   // native only
//! # let _ = rgba;
//! # }
//! ```
//!
//! On the web there is no blocking variant and there cannot be: mapping
//! completes on the event loop, so a wait that does not yield starves the
//! callback it is waiting for. Use [`copy_texture_to_buffer`] and drive the
//! mapping yourself from an async context.

/// One IEEE 754 binary16 as an `f32`.
///
/// Exact for every input, subnormals and infinities included — a `bits >> 10`
/// shortcut gets the normals right and silently flattens the subnormals to
/// zero, which is most of the interesting range when the values being read
/// back are small.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as u32;
    match exponent {
        // Subnormal (and zero): no implicit leading one, and a fixed exponent
        // of 2^-24. Writing it as a multiply rather than by renormalising into
        // float32 bits is both shorter and the version that is obviously right.
        0 => sign * mantissa as f32 * 5.960_464_5e-8,
        0x1f => {
            if mantissa == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => {
            let s = u32::from(bits & 0x8000 != 0) << 31;
            f32::from_bits(s | (((exponent + 127 - 15) as u32) << 23) | (mantissa << 13))
        }
    }
}

/// An `f32` as the nearest IEEE 754 binary16, round-to-nearest-even.
///
/// The inverse of [`f16_to_f32`], for writing a float texture from the CPU.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exp >= 0x1f {
        // Overflow, infinity, or NaN. A NaN must stay a NaN rather than
        // becoming an infinity, so keep a mantissa bit set.
        let payload = if ((bits >> 23) & 0xff) == 0xff && mantissa != 0 {
            0x200
        } else {
            0
        };
        return sign | 0x7c00 | payload;
    }
    if exp <= 0 {
        // Subnormal or zero in half precision.
        if exp < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = (m >> shift) as u16;
        let round = (m >> (shift - 1)) & 1;
        return sign | (half + round as u16);
    }
    let half = ((exp as u32) << 10) as u16 | (mantissa >> 13) as u16;
    // Round to nearest, ties to even.
    let round = (mantissa >> 12) & 1;
    let sticky = (mantissa & 0xfff != 0) as u32;
    let up = round & (sticky | (half as u32 & 1));
    sign | (half + up as u16)
}

/// The padded row stride a buffer copy of `width` texels of `format` needs.
pub fn padded_bytes_per_row(width: u32, format: wgpu::TextureFormat) -> u32 {
    let unpadded = width * bytes_per_texel(format);
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

/// Bytes one texel occupies, for the uncompressed formats this crate reads
/// back. Panics on a block-compressed format, which cannot be read this way.
pub fn bytes_per_texel(format: wgpu::TextureFormat) -> u32 {
    // `block_copy_size`, not `target_pixel_byte_cost`. The latter is what a
    // texel costs as a *render attachment*, which is padded — it reports 8 for
    // `Rgba8Unorm`, and a readback sized from it reads every row at twice the
    // stride and shears the image.
    format
        .block_copy_size(None)
        .unwrap_or_else(|| panic!("{format:?} has no fixed texel size"))
}

/// Record a copy of `texture` into `buffer`, with the row padding a buffer
/// copy requires.
///
/// The buffer must be at least `padded_bytes_per_row(w, format) * h` bytes and
/// carry `COPY_DST | MAP_READ`. Use [`unpad`] on the mapped bytes.
pub fn copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
) {
    let (width, height) = (texture.width(), texture.height());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row(width, texture.format())),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// A buffer sized for [`copy_texture_to_buffer`] from `texture`.
pub fn staging_buffer(device: &wgpu::Device, texture: &wgpu::Texture) -> wgpu::Buffer {
    let size =
        padded_bytes_per_row(texture.width(), texture.format()) as u64 * texture.height() as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threers readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

/// Drop the row padding, leaving a tightly packed image.
pub fn unpad(padded: &[u8], width: u32, height: u32, format: wgpu::TextureFormat) -> Vec<u8> {
    let row = (width * bytes_per_texel(format)) as usize;
    let stride = padded_bytes_per_row(width, format) as usize;
    let mut out = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        out.extend_from_slice(&padded[y * stride..y * stride + row]);
    }
    out
}

/// Copy a texture into host memory and wait for it. **Native only.**
///
/// Returns tightly packed bytes in the texture's own format — four per texel
/// for `Rgba8Unorm`, eight for `Rgba16Float`. Use [`as_f32`] to decode a float
/// format.
#[cfg(not(target_arch = "wasm32"))]
pub fn texture_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    let buffer = staging_buffer(device, texture);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("threers readback copy"),
    });
    copy_texture_to_buffer(&mut encoder, texture, &buffer);
    queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let mapped = slice
        .get_mapped_range()
        .expect("buffer range is mapped after a blocking poll");
    let out = unpad(&mapped, texture.width(), texture.height(), texture.format());
    drop(mapped);
    buffer.unmap();
    out
}

/// Decode `*16Float` bytes into `f32`, one entry per channel.
///
/// `channels` is how many the format has: 1 for `R16Float`, 2 for `Rg16Float`,
/// 4 for `Rgba16Float`.
pub fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// One channel of a `*16Float` texture, read back as `f32`. **Native only.**
///
/// What a test wants nine times out of ten: the divergence field, the red
/// channel of the dye, a depth buffer's one plane.
#[cfg(not(target_arch = "wasm32"))]
pub fn texture_channel_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    channel: usize,
) -> Vec<f32> {
    let bytes = texture_bytes(device, queue, texture);
    if texture.format() == wgpu::TextureFormat::Rg11b10Ufloat {
        assert!(channel < 3, "channel {channel} of a 3-channel format");
        return bytes
            .chunks_exact(4)
            .map(|t| {
                let word = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
                packed_float_to_f32(word, channel)
            })
            .collect();
    }
    let channels = (bytes_per_texel(texture.format()) / 2) as usize;
    assert!(
        channel < channels,
        "channel {channel} of a {channels}-channel format"
    );
    bytes
        .chunks_exact(channels * 2)
        .map(|texel| {
            let at = channel * 2;
            f16_to_f32(u16::from_le_bytes([texel[at], texel[at + 1]]))
        })
        .collect()
}

/// One channel of an `rg11b10ufloat` word.
///
/// Three unsigned floats in thirty-two bits: r and g are 5 bits of exponent
/// and 6 of mantissa, b is 5 and 5. No sign bit and no infinities worth
/// honouring, but the *bias is the same 15 as a half*, so each one widens to
/// a half's bit pattern by shifting the mantissa up — which is why this is a
/// shift and a reuse rather than a second decoder.
pub fn packed_float_to_f32(word: u32, channel: usize) -> f32 {
    let (bits, mantissa_bits) = match channel {
        0 => (word & 0x7ff, 6),
        1 => ((word >> 11) & 0x7ff, 6),
        _ => ((word >> 22) & 0x3ff, 5),
    };
    let exponent = bits >> mantissa_bits;
    let mantissa = bits & ((1 << mantissa_bits) - 1);
    let half = (exponent << 10) | (mantissa << (10 - mantissa_bits));
    f16_to_f32(half as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed three-channel float, against values it can hold exactly.
    ///
    /// Its exponent bias is a half's, so the decode is a shift — and a shift
    /// in the wrong direction still gives plausible-looking small numbers,
    /// which is why this checks the two channel widths separately.
    #[test]
    fn packed_floats_decode_exactly() {
        // 1.0 is exponent 15, mantissa 0 in every one of the three.
        let one = 15u32 << 6 | (15u32 << 6) << 11 | (15u32 << 5) << 22;
        for c in 0..3 {
            assert_eq!(packed_float_to_f32(one, c), 1.0, "channel {c}");
        }
        // Zero everywhere, and the largest mantissa step at exponent 15:
        // 1 + 1/64 in the 6-bit channels, 1 + 1/32 in the 5-bit one.
        assert_eq!(packed_float_to_f32(0, 0), 0.0);
        let step = (15u32 << 6 | 1) | ((15u32 << 5 | 1) << 22);
        assert_eq!(packed_float_to_f32(step, 0), 1.0 + 1.0 / 64.0);
        assert_eq!(packed_float_to_f32(step, 2), 1.0 + 1.0 / 32.0);
        // And the channels do not bleed into one another.
        assert_eq!(packed_float_to_f32(step, 1), 0.0);
    }

    /// Spot values, including the ones the copies this module replaced got
    /// wrong.
    ///
    /// Of the eight separate `half_to_f32` implementations that predated this
    /// module, **three decoded subnormals at a half or a quarter of their
    /// value** — a renormalising loop whose exponent bookkeeping started one or
    /// two steps off. One of the eight carried a comment recording that the bug
    /// had been found and fixed *there*, which is the clearest possible
    /// argument for there being one copy. Subnormal halves are everything below
    /// 6.1e-5: the deep shadow end of an EXR, and the almost-but-not-quite
    /// black of a float texture.
    #[test]
    fn half_floats_decode_exactly() {
        // Bit patterns with known values, including the two the shortcut
        // implementations get wrong.
        let cases: &[(u16, f32)] = &[
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3555, 0.333_251_95),
            (0x7bff, 65504.0),        // the largest finite half
            (0x0400, 6.103_515_6e-5), // the smallest normal
            (0x0001, 5.960_464_5e-8), // the smallest subnormal
            (0x03ff, 6.097_555e-5),   // the largest subnormal
        ];
        for &(bits, expected) in cases {
            let got = f16_to_f32(bits);
            assert!(
                (got - expected).abs() <= expected.abs() * 1e-6,
                "{bits:#06x}: {got} != {expected}"
            );
        }
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    /// Every half float, through f32 and back. A decoder that is merely
    /// plausible passes a handful of spot checks; this is all 65 536 of them.
    #[test]
    fn every_half_float_round_trips() {
        for bits in 0..=u16::MAX {
            let v = f16_to_f32(bits);
            if v.is_nan() {
                assert!(nan_half(f32_to_f16(v)), "{bits:#06x} lost its NaN");
                continue;
            }
            let back = f32_to_f16(v);
            assert_eq!(back, bits, "{bits:#06x} -> {v} -> {back:#06x}");
        }
    }

    /// Is this half-float bit pattern a NaN? Free function rather than a
    /// method, because `is_*` taking `self` by value reads as a conversion.
    fn nan_half(bits: u16) -> bool {
        (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff) != 0
    }

    #[test]
    fn rows_are_padded_to_the_alignment_and_unpadded_back() {
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        // 100 texels is 400 bytes, which is not a multiple of 256.
        assert_eq!(padded_bytes_per_row(100, fmt), 512);
        // 64 texels is exactly 256.
        assert_eq!(padded_bytes_per_row(64, fmt), 256);

        let (w, h) = (100u32, 3u32);
        let stride = padded_bytes_per_row(w, fmt) as usize;
        let mut padded = vec![0u8; stride * h as usize];
        for y in 0..h as usize {
            for x in 0..(w * 4) as usize {
                padded[y * stride + x] = (y * 7 + x) as u8;
            }
        }
        let tight = unpad(&padded, w, h, fmt);
        assert_eq!(tight.len(), (w * 4 * h) as usize);
        for y in 0..h as usize {
            for x in 0..(w * 4) as usize {
                assert_eq!(tight[y * (w * 4) as usize + x], (y * 7 + x) as u8);
            }
        }
    }

    #[test]
    fn texel_sizes_cover_what_gets_read_back() {
        use wgpu::TextureFormat as F;
        assert_eq!(bytes_per_texel(F::R8Unorm), 1);
        assert_eq!(bytes_per_texel(F::Rgba8Unorm), 4);
        assert_eq!(bytes_per_texel(F::Rgba8UnormSrgb), 4);
        assert_eq!(bytes_per_texel(F::R16Float), 2);
        assert_eq!(bytes_per_texel(F::Rg16Float), 4);
        assert_eq!(bytes_per_texel(F::Rgba16Float), 8);
        assert_eq!(bytes_per_texel(F::Rgba32Float), 16);
    }
}
