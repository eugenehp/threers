//! OpenEXR loader. Parses the EXR header (magic + version + attribute table)
//! and decodes RGB/RGBA HALF or FLOAT scanlines, either clipped to 8-bit RGBA
//! (`parse`) or kept as half-float (`parse_hdr`).
//!
//! Handles the codecs almost every EXR in the wild actually uses: `NONE`,
//! `ZIPS` (one scanline per chunk) and `ZIP` (sixteen), the last via the
//! crate's own DEFLATE decoder plus EXR's byte-interleave and delta predictor.
//! `PIZ`, `B44` and `DWA` return the matching `Unsupported` variant.

use crate::textures::{Texture, TextureFormat};

pub struct ExrLoader;

#[derive(Debug)]
pub enum ExrError {
    BadMagic,
    BadHeader,
    /// Generic fallback for `UnsupportedPiz` / `UnsupportedB44` / `UnsupportedDwa`
    /// when downstream code doesn't need the specific tag.
    UnsupportedCompression,
    /// PIZ wavelet + Huffman. Complex; needs ~1k LOC port. Not implemented.
    UnsupportedPiz,
    /// B44/B44A 16×16 fixed-rate block encoding. Not implemented.
    UnsupportedB44,
    /// DWA (DreamWorks) JPEG-like codec. Not implemented.
    UnsupportedDwa,
    UnsupportedChannel,
}

const MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

impl ExrLoader {
    /// Decode to `Rgba8Unorm`, clipping values outside `0..=1`.
    ///
    /// Fine for EXRs that are already display-referred. For anything with real
    /// range — a sky, an environment probe, a render out of a compositor — use
    /// [`parse_hdr`](Self::parse_hdr) instead: clipping a scene-referred image
    /// to 8 bits throws away most of what EXR is for.
    pub fn parse(bytes: &[u8]) -> Result<Texture, ExrError> {
        Self::decode(bytes, false)
    }

    /// Decode to `Rgba16Float`, keeping the source's range.
    ///
    /// EXR is a scene-referred format: a star map runs from a Gaia smudge at
    /// 0.001 to Sirius well past 1.0, and NASA's Deep Star Maps put over half
    /// their texels below 0.01. Squeezing that into 8 bits leaves the sky
    /// almost entirely black; keeping it as half-float lets the renderer's tone
    /// mapping decide instead.
    pub fn parse_hdr(bytes: &[u8]) -> Result<Texture, ExrError> {
        Self::decode(bytes, true)
    }

    fn decode(bytes: &[u8], hdr: bool) -> Result<Texture, ExrError> {
        if bytes.len() < 8 || bytes[0..4] != MAGIC {
            return Err(ExrError::BadMagic);
        }
        // version: u32 LE — low byte version, high 24 bits = flags.
        let mut pos = 8usize;
        // Attribute table: triples of (name\0, type\0, size: u32 LE, data...)
        // until a single null byte.
        let mut data_window: Option<[i32; 4]> = None;
        let mut compression: u8 = 0; // 0 == NONE
        let mut channels_present: Vec<(String, u8)> = Vec::new(); // (name, pixel_type)
        loop {
            if pos >= bytes.len() {
                return Err(ExrError::BadHeader);
            }
            if bytes[pos] == 0 {
                pos += 1;
                break;
            }
            // Read attribute name (zero-terminated).
            let name_start = pos;
            while pos < bytes.len() && bytes[pos] != 0 {
                pos += 1;
            }
            if pos >= bytes.len() {
                return Err(ExrError::BadHeader);
            }
            let name = String::from_utf8_lossy(&bytes[name_start..pos]).to_string();
            pos += 1;
            // Type.
            let type_start = pos;
            while pos < bytes.len() && bytes[pos] != 0 {
                pos += 1;
            }
            if pos >= bytes.len() {
                return Err(ExrError::BadHeader);
            }
            let attr_type = String::from_utf8_lossy(&bytes[type_start..pos]).to_string();
            pos += 1;
            // Size.
            if pos + 4 > bytes.len() {
                return Err(ExrError::BadHeader);
            }
            let size =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            pos += 4;
            if pos + size > bytes.len() {
                return Err(ExrError::BadHeader);
            }
            let data = &bytes[pos..pos + size];
            pos += size;
            match name.as_str() {
                "dataWindow" if attr_type == "box2i" && size >= 16 => {
                    let read_i =
                        |o| i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    data_window = Some([read_i(0), read_i(4), read_i(8), read_i(12)]);
                }
                "compression" if attr_type == "compression" && size >= 1 => {
                    compression = data[0];
                }
                "channels" if attr_type == "chlist" => {
                    let mut k = 0usize;
                    while k < size {
                        if data[k] == 0 {
                            break;
                        }
                        let n_start = k;
                        while k < size && data[k] != 0 {
                            k += 1;
                        }
                        let cname = String::from_utf8_lossy(&data[n_start..k]).to_string();
                        k += 1;
                        if k + 16 > size {
                            break;
                        }
                        let pixel_type = data[k];
                        k += 16;
                        channels_present.push((cname, pixel_type));
                    }
                }
                _ => {}
            }
        }
        // 0 = NONE, 1 = RLE (TODO), 2 = ZIPS, 3 = ZIP, 4 = PIZ,
        // 5 = PXR24, 6 = B44, 7 = B44A, 8 = DWAA, 9 = DWAB.
        match compression {
            0 | 2 | 3 => {}
            4 => return Err(ExrError::UnsupportedPiz),
            6 | 7 => return Err(ExrError::UnsupportedB44),
            8 | 9 => return Err(ExrError::UnsupportedDwa),
            _ => return Err(ExrError::UnsupportedCompression),
        }
        let Some(dw) = data_window else {
            return Err(ExrError::BadHeader);
        };
        let width = (dw[2] - dw[0] + 1).max(0) as usize;
        let height = (dw[3] - dw[1] + 1).max(0) as usize;
        if width == 0 || height == 0 {
            return Err(ExrError::BadHeader);
        }

        // Determine RGB(A) channel layout.
        let pixel_type_of = |name: &str| -> Option<u8> {
            channels_present
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| *t)
        };
        let r_type = pixel_type_of("R").ok_or(ExrError::UnsupportedChannel)?;
        let g_type = pixel_type_of("G").ok_or(ExrError::UnsupportedChannel)?;
        let b_type = pixel_type_of("B").ok_or(ExrError::UnsupportedChannel)?;
        let a_type = pixel_type_of("A");

        // ZIP packs 16 scanlines into one chunk; NONE and ZIPS store one each.
        let scanlines_per_block = if compression == 3 { 16 } else { 1 };

        // Skip the chunk-offsets table: one u64 per *chunk*, not per scanline.
        // Assuming one per scanline overshoots a ZIP file by 15/16ths of the
        // table and lands mid-pixel-data, which then fails as a bad header.
        let offsets_size = height.div_ceil(scanlines_per_block) * 8;
        if pos + offsets_size > bytes.len() {
            return Err(ExrError::BadHeader);
        }
        pos += offsets_size;

        // For each scanline: y(i32) + size(u32) + interleaved channel data.
        let sample_size = |t: u8| -> usize {
            match t {
                1 => 2,
                2 => 4,
                _ => 0,
            }
        };
        let read_sample = |t: u8, data: &[u8], off: usize| -> f32 {
            match t {
                1 => f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]])),
                2 => f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]),
                _ => 0.0,
            }
        };

        let bpp = if hdr { 8 } else { 4 };

        // Channel layout is fixed for the whole image, so work it out once. It
        // used to be re-derived per scanline — a sort of the channel list per
        // chunk and a `HashMap` with `String` keys built and thrown away for
        // every one of four thousand rows.
        let mut sorted_chans: Vec<&(String, u8)> = channels_present.iter().collect();
        sorted_chans.sort_by(|a, b| a.0.cmp(&b.0));
        let bytes_per_scanline: usize = sorted_chans
            .iter()
            .map(|(_, t)| sample_size(*t) * width)
            .sum();
        let offset_of = |want: &str| -> usize {
            let mut cursor = 0usize;
            for (name, t) in &sorted_chans {
                if name == want {
                    return cursor;
                }
                cursor += sample_size(*t) * width;
            }
            0
        };
        let (r_off, g_off, b_off, a_off) = (
            offset_of("R"),
            offset_of("G"),
            offset_of("B"),
            offset_of("A"),
        );

        // Walk the block headers first, without decompressing. Each chunk is an
        // independent deflate stream covering a known band of scanlines, so
        // once their extents are known they can be decoded in parallel — and
        // decompression is the whole cost here.
        let mut blocks: Vec<&[u8]> = Vec::new();
        {
            let mut scan = pos;
            let mut y = 0usize;
            while y < height {
                if scan + 8 > bytes.len() {
                    return Err(ExrError::BadHeader);
                }
                let block_size = u32::from_le_bytes([
                    bytes[scan + 4],
                    bytes[scan + 5],
                    bytes[scan + 6],
                    bytes[scan + 7],
                ]) as usize;
                scan += 8;
                if scan + block_size > bytes.len() {
                    return Err(ExrError::BadHeader);
                }
                blocks.push(&bytes[scan..scan + block_size]);
                scan += block_size;
                y += scanlines_per_block;
            }
        }

        let mut rgba: Vec<u8> = vec![0u8; width * height * bpp];
        let chunk_bytes = scanlines_per_block * width * bpp;
        let decode_block = |block_compressed: &[u8], out: &mut [u8], first_y: usize| {
            let block: Vec<u8> = if compression == 0 {
                block_compressed.to_vec()
            } else {
                match super::deflate::inflate_zlib(block_compressed) {
                    Ok(v) => exr_reorder(&v),
                    Err(_) => return,
                }
            };
            let scanline_count = scanlines_per_block.min(height - first_y);
            for s in 0..scanline_count {
                if (s + 1) * bytes_per_scanline > block.len() {
                    break;
                }
                let scanline = &block[s * bytes_per_scanline..(s + 1) * bytes_per_scanline];
                let row = &mut out[s * width * bpp..(s + 1) * width * bpp];
                for x in 0..width {
                    let r = read_sample(r_type, scanline, r_off + x * sample_size(r_type));
                    let g = read_sample(g_type, scanline, g_off + x * sample_size(g_type));
                    let b = read_sample(b_type, scanline, b_off + x * sample_size(b_type));
                    let a = match a_type {
                        Some(t) => read_sample(t, scanline, a_off + x * sample_size(t)),
                        None => 1.0,
                    };
                    let o = x * bpp;
                    if hdr {
                        for (i, v) in [r, g, b, a].iter().enumerate() {
                            row[o + i * 2..o + i * 2 + 2]
                                .copy_from_slice(&f32_to_f16(*v).to_le_bytes());
                        }
                    } else {
                        let clip = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                        row[o..o + 4].copy_from_slice(&[clip(r), clip(g), clip(b), clip(a)]);
                    }
                }
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(blocks.len().max(1))
                .max(1);
            let per = blocks.len().div_ceil(threads);
            let blocks = &blocks;
            let decode_block = &decode_block;
            std::thread::scope(|scope| {
                for (band, out) in rgba.chunks_mut(per * chunk_bytes).enumerate() {
                    scope.spawn(move || {
                        for (i, slot) in out.chunks_mut(chunk_bytes).enumerate() {
                            let idx = band * per + i;
                            if idx < blocks.len() {
                                decode_block(blocks[idx], slot, idx * scanlines_per_block);
                            }
                        }
                    });
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        for (idx, slot) in rgba.chunks_mut(chunk_bytes).enumerate() {
            if idx < blocks.len() {
                decode_block(blocks[idx], slot, idx * scanlines_per_block);
            }
        }

        if hdr {
            return Ok(Texture::new(
                width as u32,
                height as u32,
                TextureFormat::Rgba16Float,
                rgba,
            ));
        }
        Ok(Texture::new(
            width as u32,
            height as u32,
            TextureFormat::Rgba8Unorm,
            rgba,
        ))
    }
}

/// Undo the two byte-level transforms EXR applies before deflating.
///
/// The encoder splits the block into even and odd bytes and *then* delta-codes
/// the result, so decoding has to undo them in the opposite order: predictor
/// first, over the still-split buffer, and only then reassemble. Doing it the
/// other way round decompresses cleanly and produces pure noise, which is the
/// failure this ordering exists to avoid.
fn exr_reorder(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    // 1. Reverse the up-delta predictor: each byte carries its difference from
    //    the one before, biased by 128.
    let mut buf = input.to_vec();
    for i in 1..buf.len() {
        buf[i] = buf[i].wrapping_add(buf[i - 1]).wrapping_sub(128);
    }

    // 2. Interleave the halves back together: the first half holds the bytes
    //    that go to even positions, the second half the odd ones.
    let half = buf.len().div_ceil(2);
    let mut out = vec![0u8; buf.len()];
    let (mut lo, mut hi) = (0usize, half);
    for (i, slot) in out.iter_mut().enumerate() {
        if i % 2 == 0 {
            *slot = buf[lo];
            lo += 1;
        } else {
            *slot = buf[hi];
            hi += 1;
        }
    }
    out
}

/// IEEE 754 half-precision → single-precision conversion.
/// The crate's one f16 → f32, for the same reason as its inverse below.
fn f16_to_f32(h: u16) -> f32 {
    crate::renderer::gpu_texture::f16_bits_to_f32(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap `payload` in a zlib stream of stored (uncompressed) DEFLATE blocks.
    /// The Adler trailer is not checked on the way back in, so zeroes do.
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        for (i, chunk) in payload.chunks(0xffff).enumerate() {
            let last = (i + 1) * 0xffff >= payload.len();
            out.push(if last { 0x01 } else { 0x00 }); // BFINAL, BTYPE=stored
            let len = chunk.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&(!len).to_le_bytes());
            out.extend_from_slice(chunk);
        }
        out.extend_from_slice(&[0, 0, 0, 0]); // Adler-32 placeholder
        out
    }

    /// The encoder side of [`exr_reorder`]: split into even/odd halves, then
    /// delta-code. Decoding has to undo these in the opposite order.
    fn exr_pack(raw: &[u8]) -> Vec<u8> {
        let half = raw.len().div_ceil(2);
        let mut split = vec![0u8; raw.len()];
        let (mut lo, mut hi) = (0usize, half);
        for (i, &b) in raw.iter().enumerate() {
            if i % 2 == 0 {
                split[lo] = b;
                lo += 1;
            } else {
                split[hi] = b;
                hi += 1;
            }
        }
        let mut out = split.clone();
        for i in (1..out.len()).rev() {
            out[i] = split[i].wrapping_sub(split[i - 1]).wrapping_add(128);
        }
        out
    }

    /// A minimal single-part scanline EXR: 3 HALF channels, given compression.
    fn synth_exr(width: usize, height: usize, compression: u8) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(&MAGIC);
        f.extend_from_slice(&2u32.to_le_bytes());

        let attr = |name: &str, ty: &str, data: &[u8], f: &mut Vec<u8>| {
            f.extend_from_slice(name.as_bytes());
            f.push(0);
            f.extend_from_slice(ty.as_bytes());
            f.push(0);
            f.extend_from_slice(&(data.len() as u32).to_le_bytes());
            f.extend_from_slice(data);
        };

        // chlist: alphabetical, each entry name\0 + pixelType i32 + pLinear +
        // 3 reserved + xSampling i32 + ySampling i32.
        let mut chlist = Vec::new();
        for name in ["B", "G", "R"] {
            chlist.extend_from_slice(name.as_bytes());
            chlist.push(0);
            chlist.extend_from_slice(&1i32.to_le_bytes()); // HALF
            chlist.extend_from_slice(&[0, 0, 0, 0]);
            chlist.extend_from_slice(&1i32.to_le_bytes());
            chlist.extend_from_slice(&1i32.to_le_bytes());
        }
        chlist.push(0);
        attr("channels", "chlist", &chlist, &mut f);
        attr("compression", "compression", &[compression], &mut f);
        let mut dw = Vec::new();
        for v in [0i32, 0, width as i32 - 1, height as i32 - 1] {
            dw.extend_from_slice(&v.to_le_bytes());
        }
        attr("dataWindow", "box2i", &dw, &mut f);
        f.push(0); // end of header

        let per_block = if compression == 3 { 16 } else { 1 };
        let chunks = height.div_ceil(per_block);
        // Chunk offset table — one u64 per chunk. The parser skips it, so the
        // values do not matter, but the *count* is exactly what this tests.
        f.extend_from_slice(&vec![0u8; chunks * 8]);

        // B = 0.0, G = 0.5, R = 1.0 as IEEE half, channels in sorted order.
        let scanline: Vec<u8> = [0x0000u16, 0x3800, 0x3c00]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, width))
            .flat_map(|v| v.to_le_bytes())
            .collect();
        for c in 0..chunks {
            let lines = per_block.min(height - c * per_block);
            let raw: Vec<u8> = scanline
                .iter()
                .cycle()
                .take(scanline.len() * lines)
                .copied()
                .collect();
            let payload = if compression == 0 {
                raw
            } else {
                zlib_stored(&exr_pack(&raw))
            };
            f.extend_from_slice(&((c * per_block) as i32).to_le_bytes());
            f.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            f.extend_from_slice(&payload);
        }
        f
    }

    #[test]
    fn zip_compressed_scanlines_decode() {
        // ZIP is what almost every EXR in the wild uses — NASA's Deep Star Maps
        // included. The offset table holds one entry per 16-scanline *chunk*;
        // reading it as one per scanline overshoots by 15/16ths of the table,
        // lands mid-payload, and the file fails as a bad header.
        let (w, h) = (4usize, 32usize);
        let tex = ExrLoader::parse(&synth_exr(w, h, 3)).expect("ZIP EXR should decode");
        assert_eq!((tex.width, tex.height), (w as u32, h as u32));
        assert_eq!(tex.data.len(), w * h * 4);
        // …and the pixels are the ones that went in. Getting the byte
        // transforms back in the wrong order still decompresses and still
        // fills the buffer — with noise.
        assert!(
            tex.data.chunks_exact(4).all(|p| p == [255, 128, 0, 255]),
            "ZIP payload decoded to the wrong values: {:?}",
            &tex.data[..8]
        );
    }

    #[test]
    fn uncompressed_and_zips_still_decode() {
        for compression in [0u8, 2] {
            let tex = ExrLoader::parse(&synth_exr(4, 8, compression))
                .unwrap_or_else(|e| panic!("compression {compression} failed: {e:?}"));
            assert_eq!((tex.width, tex.height), (4, 8));
            assert_eq!(tex.data.len(), 4 * 8 * 4);
            assert!(tex.data.chunks_exact(4).all(|p| p == [255, 128, 0, 255]));
        }
    }

    #[test]
    fn a_partial_final_chunk_is_handled() {
        // 20 scanlines is one full ZIP chunk plus a 4-line remainder.
        let tex = ExrLoader::parse(&synth_exr(4, 20, 3)).expect("ragged height should decode");
        assert_eq!(tex.height, 20);
        assert_eq!(tex.data.len(), 4 * 20 * 4);
    }

    #[test]
    fn unsupported_codecs_say_which_one() {
        assert!(matches!(
            ExrLoader::parse(&synth_exr(4, 4, 4)),
            Err(ExrError::UnsupportedPiz)
        ));
        assert!(matches!(
            ExrLoader::parse(&synth_exr(4, 4, 8)),
            Err(ExrError::UnsupportedDwa)
        ));
    }
}

/// Single-precision → IEEE 754 half-precision, round-to-nearest-even.
///
/// Values past half's ~65504 ceiling saturate to infinity rather than wrap;
/// subnormals below ~6e-8 flush to zero.
/// The crate's one f32 → f16. Kept here as a name so the decoder above reads
/// symmetrically with [`f16_to_f32`]; the implementation is shared, because two
/// copies of this is how one of them spent a long time silently discarding
/// every HDR value above 2.0 while the other was fine.
fn f32_to_f16(v: f32) -> u16 {
    crate::renderer::gpu_texture::f32_to_f16_bits(v)
}

#[cfg(test)]
mod half_tests {
    use super::{f16_to_f32, f32_to_f16};

    #[test]
    fn half_round_trips_representable_values() {
        for v in [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0,
            -65504.0,
            6.103_515_6e-5,
        ] {
            let back = f16_to_f32(f32_to_f16(v));
            assert_eq!(back, v, "{v} did not survive the round trip");
        }
    }

    #[test]
    fn half_saturates_and_flushes_instead_of_wrapping() {
        // Past half's ceiling, saturate — wrapping would turn the brightest
        // star in a sky map into a black hole.
        assert!(f16_to_f32(f32_to_f16(1.0e30)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(-1.0e30)).is_infinite());
        // Below the subnormal floor, flush to zero with the sign kept.
        assert_eq!(f16_to_f32(f32_to_f16(1.0e-12)), 0.0);
    }

    #[test]
    fn small_values_keep_their_precision() {
        // This is the whole reason for the HDR path: a star map's faint end.
        for v in [1.0e-3f32, 5.0e-3, 1.0e-2, 0.1] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!(
                (back - v).abs() / v < 1.0e-2,
                "{v} came back as {back}, {:.1}% off",
                (back - v).abs() / v * 100.0
            );
        }
    }
}
