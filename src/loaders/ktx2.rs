//! KTX2 — the container glTF uses for GPU-compressed textures.
//!
//! `KHR_texture_basisu` puts textures in a `.ktx2`, and an ARKit `.usdz` leans
//! on compressed textures too. The decode target already existed here: the
//! crate's [`TextureFormat`] carries block-compressed data with a mip chain.
//! What was missing was the container.
//!
//! # Layout
//!
//! A 12-byte identifier, a fixed header, an index of mip levels, then three
//! optional blocks the header points at:
//!
//! | block | what it says |
//! |---|---|
//! | Data Format Descriptor | the colour model and transfer function |
//! | Key/Value Data | `KTXorientation`, `KTXwriter`, and anything else |
//! | Supercompression Global Data | the Basis codebooks, for `BASISLZ` |
//!
//! Levels are indexed **largest first** in the file, which is the order this
//! crate's mip chain wants, so no reversal is needed — but the index gives each
//! level an absolute offset, so they need not be contiguous and must not be
//! assumed to be.
//!
//! # What is not here
//!
//! Basis Universal transcoding. A `BASISLZ` or `UASTC` payload has to be
//! transcoded to whatever the GPU takes, and that is a codec rather than a
//! container — ETC1S has its own entropy coder and codebooks. Such a file is
//! refused by name rather than returned as noise. `ZSTD` supercompression is
//! likewise refused: this crate has an inflate and no zstd.

use std::collections::HashMap;
use std::sync::Arc;

use crate::textures::{Texture, TextureFormat};

const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

#[derive(Debug, PartialEq)]
pub enum Ktx2Error {
    /// The 12-byte identifier is not there.
    NotKtx2,
    /// The file ended inside a block it declared.
    Truncated,
    /// Structurally KTX2 and internally inconsistent.
    Corrupt,
    /// A format or supercompression this crate cannot turn into a texture,
    /// named so a caller knows what is missing rather than merely that
    /// something is.
    Unsupported(&'static str),
}

impl std::fmt::Display for Ktx2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ktx2Error::NotKtx2 => write!(f, "not a KTX2 file"),
            Ktx2Error::Truncated => write!(f, "truncated KTX2 file"),
            Ktx2Error::Corrupt => write!(f, "corrupt KTX2 file"),
            Ktx2Error::Unsupported(what) => write!(f, "unsupported KTX2 texture: {what}"),
        }
    }
}

impl std::error::Error for Ktx2Error {}

/// How the mip level payloads were compressed on top of their pixel format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Supercompression {
    None,
    /// Basis Universal's own scheme, with codebooks in the global data.
    BasisLz,
    Zstd,
    Zlib,
}

/// One mip level, as the index describes it.
#[derive(Debug, Clone)]
pub struct Level {
    /// The bytes as stored — still supercompressed, if the file is.
    pub data: Vec<u8>,
    /// What they come to once decompressed. Equal to `data.len()` when there is
    /// no supercompression.
    pub uncompressed_length: usize,
}

/// A parsed KTX2 container.
#[derive(Debug, Clone)]
pub struct Ktx2 {
    /// The Vulkan format, or 0 for a Basis payload whose real format is in the
    /// data format descriptor.
    pub vk_format: u32,
    pub type_size: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub faces: u32,
    /// Zero means "only the base level, and generate the rest at load time",
    /// which is not the same as one.
    pub level_count: u32,
    pub supercompression: Supercompression,
    /// Largest first.
    pub levels: Vec<Level>,
    /// `KHR_DF_TRANSFER_SRGB` is 2; 1 is linear. For a Basis payload this is
    /// the only thing that says which.
    pub transfer_function: u8,
    pub key_value: HashMap<String, Vec<u8>>,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, Ktx2Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, Ktx2Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Ktx2Error> {
        let end = self.at.checked_add(n).ok_or(Ktx2Error::Truncated)?;
        let s = self.data.get(self.at..end).ok_or(Ktx2Error::Truncated)?;
        self.at = end;
        Ok(s)
    }
}

/// Parse a `.ktx2`.
pub fn parse(bytes: &[u8]) -> Result<Ktx2, Ktx2Error> {
    if bytes.len() < IDENTIFIER.len() || bytes[..IDENTIFIER.len()] != IDENTIFIER {
        return Err(Ktx2Error::NotKtx2);
    }
    let mut r = Reader {
        data: bytes,
        at: IDENTIFIER.len(),
    };
    let vk_format = r.u32()?;
    let type_size = r.u32()?;
    let width = r.u32()?;
    let height = r.u32()?;
    let depth = r.u32()?;
    let layers = r.u32()?;
    let faces = r.u32()?;
    let level_count = r.u32()?;
    let supercompression = match r.u32()? {
        0 => Supercompression::None,
        1 => Supercompression::BasisLz,
        2 => Supercompression::Zstd,
        3 => Supercompression::Zlib,
        _ => return Err(Ktx2Error::Corrupt),
    };
    let dfd_offset = r.u32()? as usize;
    let dfd_length = r.u32()? as usize;
    let kvd_offset = r.u32()? as usize;
    let kvd_length = r.u32()? as usize;
    let _sgd_offset = r.u64()?;
    let _sgd_length = r.u64()?;

    if width == 0 || faces == 0 {
        return Err(Ktx2Error::Corrupt);
    }

    // The level index. `levelCount = 0` still has one level in the file.
    let stored_levels = level_count.max(1) as usize;
    // Each entry is three 64-bit numbers, so a count beyond the file is
    // corruption rather than a very large texture.
    if stored_levels.saturating_mul(24) > bytes.len() {
        return Err(Ktx2Error::Corrupt);
    }
    let mut levels = Vec::with_capacity(stored_levels);
    for _ in 0..stored_levels {
        let offset = r.u64()? as usize;
        let length = r.u64()? as usize;
        let uncompressed_length = r.u64()? as usize;
        let end = offset.checked_add(length).ok_or(Ktx2Error::Truncated)?;
        let data = bytes.get(offset..end).ok_or(Ktx2Error::Truncated)?.to_vec();
        levels.push(Level {
            data,
            uncompressed_length,
        });
    }

    // The data format descriptor. Only the transfer function is read: for an
    // explicit `vkFormat` the format already says whether it is sRGB, and for a
    // Basis payload this is the only place it is written.
    let transfer_function = if dfd_length >= 16 {
        bytes.get(dfd_offset + 15).copied().unwrap_or(0)
    } else {
        0
    };

    let key_value = key_values(bytes, kvd_offset, kvd_length);
    Ok(Ktx2 {
        vk_format,
        type_size,
        width,
        height,
        depth,
        layers,
        faces,
        level_count,
        supercompression,
        levels,
        transfer_function,
        key_value,
    })
}

/// The key/value block: a length, then `key\0value`, padded to four bytes.
///
/// A malformed entry ends the walk rather than failing the file — the metadata
/// is advisory, and refusing a whole texture over an unreadable `KTXwriter`
/// string would be the wrong trade.
fn key_values(bytes: &[u8], offset: usize, length: usize) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    let Some(block) = bytes.get(offset..offset + length) else {
        return out;
    };
    let mut at = 0usize;
    while at + 4 <= block.len() {
        let size = u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
            as usize;
        at += 4;
        let Some(entry) = block.get(at..at + size) else {
            break;
        };
        if let Some(split) = entry.iter().position(|b| *b == 0) {
            let key = String::from_utf8_lossy(&entry[..split]).into_owned();
            out.insert(key, entry[split + 1..].to_vec());
        }
        // Entries are aligned to four bytes.
        at += size.next_multiple_of(4);
    }
    out
}

impl Ktx2 {
    /// The Vulkan format as one of this crate's, if it has an equivalent.
    ///
    /// The block-compressed variants here are all sRGB, so a *linear* BC
    /// payload — which is what a normal map is stored as — has no home and is
    /// refused rather than labelled sRGB. Reading a normal map through an sRGB
    /// curve is wrong in a way that looks like bad lighting rather than like a
    /// bug, which is exactly the failure worth refusing.
    pub fn texture_format(&self) -> Result<TextureFormat, Ktx2Error> {
        Ok(match self.vk_format {
            37 => TextureFormat::Rgba8Unorm,       // R8G8B8A8_UNORM
            43 => TextureFormat::Rgba8UnormSrgb,   // R8G8B8A8_SRGB
            // B8G8R8A8_UNORM / _SRGB: swizzled to RGBA by `to_texture`.
            44 => TextureFormat::Rgba8Unorm,
            50 => TextureFormat::Rgba8UnormSrgb,
            9 => TextureFormat::R8Unorm,           // R8_UNORM
            97 => TextureFormat::Rgba16Float,      // R16G16B16A16_SFLOAT
            134 => TextureFormat::Bc1RgbaUnormSrgb, // BC1_RGBA_SRGB_BLOCK
            146 => TextureFormat::Bc7RgbaUnormSrgb, // BC7_SRGB_BLOCK
            131 | 133 => return Err(Ktx2Error::Unsupported("linear BC1 (no linear BC format here)")),
            145 => return Err(Ktx2Error::Unsupported("linear BC7 (no linear BC format here)")),
            0 => return Err(Ktx2Error::Unsupported("Basis Universal, which needs transcoding")),
            _ => return Err(Ktx2Error::Unsupported("this Vulkan format")),
        })
    }

    /// Every level's bytes, decompressed if the file is supercompressed.
    pub fn level_data(&self) -> Result<Vec<Vec<u8>>, Ktx2Error> {
        let mut out = Vec::with_capacity(self.levels.len());
        for level in &self.levels {
            let bytes = match self.supercompression {
                Supercompression::None => level.data.clone(),
                Supercompression::Zlib => {
                    let inflated = crate::loaders::deflate::inflate_zlib(&level.data)
                        .map_err(|_| Ktx2Error::Corrupt)?;
                    // The index says what it should come to, and a mismatch
                    // means the wrong bytes, not merely fewer.
                    if inflated.len() != level.uncompressed_length {
                        return Err(Ktx2Error::Corrupt);
                    }
                    inflated
                }
                Supercompression::Zstd => {
                    return Err(Ktx2Error::Unsupported("zstd supercompression"))
                }
                Supercompression::BasisLz => {
                    return Err(Ktx2Error::Unsupported("BasisLZ supercompression"))
                }
            };
            out.push(bytes);
        }
        Ok(out)
    }

    /// The whole thing as one texture, mip chain included.
    ///
    /// Levels come out of the file largest first, which is the order the mip
    /// chain wants, so `mips` holds levels 1..n and the base level is the
    /// texture's own data.
    pub fn to_texture(&self) -> Result<Texture, Ktx2Error> {
        let format = self.texture_format()?;
        if self.faces != 1 {
            return Err(Ktx2Error::Unsupported("cube maps"));
        }
        if self.layers > 1 {
            return Err(Ktx2Error::Unsupported("array textures"));
        }
        let mut data = self.level_data()?;
        if data.is_empty() {
            return Err(Ktx2Error::Corrupt);
        }
        if matches!(self.vk_format, 44 | 50) {
            // BGRA in the file, RGBA in the texture.
            for level in &mut data {
                for px in level.chunks_exact_mut(4) {
                    px.swap(0, 2);
                }
            }
        }
        let base = data.remove(0);
        let height = self.height.max(1);
        // The base level has to be the size the format says for these
        // dimensions, or the upload reads past it.
        if base.len() < format.data_len(self.width, height) {
            return Err(Ktx2Error::Corrupt);
        }
        let mut texture = Texture::new(self.width, height, format, base);
        // Supplied mips are already the right way up, so no flip on upload —
        // a block-compressed chain cannot be flipped by reversing rows anyway.
        texture.flip_y = false;
        texture.mips = data.into_iter().map(Arc::new).collect();
        Ok(texture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA8: &[u8] = include_bytes!("testdata/ktx/rgba8.ktx2");
    const BC7: &[u8] = include_bytes!("testdata/ktx/bc7.ktx2");
    const MIPS: &[u8] = include_bytes!("testdata/ktx/mips.ktx2");
    const ZLIB: &[u8] = include_bytes!("testdata/ktx/zlib.ktx2");
    const KV: &[u8] = include_bytes!("testdata/ktx/kv.ktx2");

    /// An uncompressed texture, read as the container says.
    ///
    /// Every expectation here is what `ktx-parse` — an independent
    /// implementation — reports for the same bytes: `vkFormat` 43, 4x4, one
    /// level of 64 bytes.
    #[test]
    fn an_uncompressed_texture_reads() {
        let k = parse(RGBA8).expect("parses");
        assert_eq!(k.vk_format, 43, "R8G8B8A8_SRGB");
        assert_eq!((k.width, k.height), (4, 4));
        assert_eq!(k.level_count, 1);
        assert_eq!(k.faces, 1);
        assert_eq!(k.supercompression, Supercompression::None);
        assert_eq!(k.levels.len(), 1);
        assert_eq!(k.levels[0].data.len(), 64, "4 * 4 * 4 bytes");
        assert_eq!(k.levels[0].uncompressed_length, 64);

        let texture = k.to_texture().expect("a texture");
        assert_eq!(texture.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!((texture.width, texture.height), (4, 4));
        // The pixels are the ones written: first pixel (0, 255, 64, 255).
        assert_eq!(&texture.data[..4], &[0, 255, 64, 255]);
        // ...and the last, fifteen steps along.
        assert_eq!(&texture.data[60..64], &[240, 15, 64, 255]);
    }

    /// A block-compressed texture keeps its blocks.
    ///
    /// The payload must not be touched: 8x8 of BC7 is four 16-byte blocks, and
    /// anything that tried to treat it as pixels would change its length.
    #[test]
    fn a_block_compressed_texture_reads() {
        let k = parse(BC7).expect("parses");
        assert_eq!(k.vk_format, 146, "BC7_SRGB_BLOCK");
        assert_eq!((k.width, k.height), (8, 8));
        let texture = k.to_texture().expect("a texture");
        assert_eq!(texture.format, TextureFormat::Bc7RgbaUnormSrgb);
        assert!(texture.format.is_block_compressed());
        assert_eq!(texture.data.len(), 64, "2x2 blocks of 16 bytes");
        assert_eq!(texture.data[0], 0x40, "the first block is untouched");
        // A block-compressed chain cannot be flipped by reversing rows, so it
        // must arrive the right way up and say so.
        assert!(!texture.flip_y);
    }

    /// A mip chain arrives largest first, with the base level separate.
    #[test]
    fn a_mip_chain_reads_in_order() {
        let k = parse(MIPS).expect("parses");
        assert_eq!(k.level_count, 4);
        assert_eq!(k.levels.len(), 4);
        // The generator filled each level with its own edge length.
        let sizes: Vec<usize> = k.levels.iter().map(|l| l.data.len()).collect();
        assert_eq!(sizes, [256, 64, 16, 4], "8x8, 4x4, 2x2, 1x1 in RGBA8");

        let texture = k.to_texture().expect("a texture");
        assert_eq!(texture.data.len(), 256, "the base level");
        assert_eq!(texture.mips.len(), 3, "and three smaller ones");
        assert_eq!(texture.data[0], 8, "level 0 was filled with 8");
        assert_eq!(texture.mips[0][0], 4, "level 1 with 4");
        assert_eq!(texture.mips[2][0], 1, "level 3 with 1");
    }

    /// Zlib supercompression is undone, and checked against the length the
    /// index declared.
    #[test]
    fn zlib_supercompression_is_decompressed() {
        let k = parse(ZLIB).expect("parses");
        assert_eq!(k.supercompression, Supercompression::Zlib);
        assert_eq!(k.levels[0].data.len(), 71, "as stored");
        assert_eq!(k.levels[0].uncompressed_length, 64, "as it will be");

        let texture = k.to_texture().expect("a texture");
        assert_eq!(texture.data.len(), 64);
        // The same pixels as the uncompressed twin, which is the point.
        let plain = parse(RGBA8).unwrap().to_texture().unwrap();
        assert_eq!(texture.data, plain.data, "compressed and plain disagree");
    }

    /// Key/value metadata is read, including the orientation every real file
    /// carries.
    #[test]
    fn key_value_metadata_reads() {
        let k = parse(KV).expect("parses");
        assert_eq!(
            k.key_value.get("KTXorientation").map(|v| v.as_slice()),
            Some(&b"rd\0"[..]),
            "{:?}",
            k.key_value.keys().collect::<Vec<_>>()
        );
        assert!(k.key_value.contains_key("KTXwriter"));
    }

    /// What cannot be turned into a texture says which thing it was.
    ///
    /// "Unsupported" on its own leaves a caller guessing; naming Basis or zstd
    /// tells them what to reach for.
    #[test]
    fn unsupported_formats_name_themselves() {
        let mut k = parse(RGBA8).unwrap();
        k.vk_format = 0;
        assert!(matches!(
            k.texture_format(),
            Err(Ktx2Error::Unsupported(what)) if what.contains("Basis")
        ));
        k.vk_format = 145;
        assert!(matches!(
            k.texture_format(),
            Err(Ktx2Error::Unsupported(what)) if what.contains("linear BC7")
        ));
        let mut z = parse(ZLIB).unwrap();
        z.supercompression = Supercompression::Zstd;
        assert!(matches!(
            z.level_data(),
            Err(Ktx2Error::Unsupported(what)) if what.contains("zstd")
        ));
    }

    /// Something that is not KTX2 is refused, and truncation never panics.
    #[test]
    fn a_non_ktx2_file_is_refused_and_truncation_is_safe() {
        assert_eq!(parse(b"not a texture").unwrap_err(), Ktx2Error::NotKtx2);
        for cut in 0..RGBA8.len() {
            let _ = parse(&RGBA8[..cut]);
        }
        for cut in 0..MIPS.len() {
            let _ = parse(&MIPS[..cut]).map(|k| k.to_texture());
        }
    }

    /// `B8G8R8A8` files load as RGBA, every mip swizzled.
    #[test]
    fn bgra_is_swizzled_to_rgba() {
        let mut k = parse(RGBA8).unwrap();
        let original = k.to_texture().unwrap();
        k.vk_format = 44;
        let swizzled = k.to_texture().unwrap();
        assert_eq!(swizzled.format, TextureFormat::Rgba8Unorm);
        let (a, b) = (&original.data[..], &swizzled.data[..]);
        assert_eq!(a.len(), b.len());
        for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            assert_eq!([p[2], p[1], p[0], p[3]], [q[0], q[1], q[2], q[3]]);
        }
    }
}
