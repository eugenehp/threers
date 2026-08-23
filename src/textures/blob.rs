//! A file for a compressed texture and its mip chain.
//!
//! PNG cannot carry BC1 blocks and has no notion of a mip chain, so a planet
//! tile encoded once offline needs somewhere to live. This is the smallest
//! container that does the job honestly: a header naming the size and format,
//! then the levels back to back, largest first.
//!
//! It is deliberately not a general image format. There is no compression of
//! the container itself — the payload is already compressed — no metadata, and
//! no attempt at forward compatibility beyond refusing to read something it
//! does not understand. The failure it is built to prevent is the silent one:
//! a stale tile whose header says 14400x14400 while the bytes are from a
//! 12720-wide crop samples as garbage rather than erroring, so every level's
//! length is checked against what the format says it must be.
//!
//! ```no_run
//! use threers::textures::{bc1, blob};
//! # let (rgba, w, h) = (vec![0u8; 64 * 64 * 4], 64u32, 64u32);
//! let (level0, mips) = bc1::encode_with_mips(&rgba, w, h);
//! blob::save("tile.tex", w, h, threers::TextureFormat::Bc1RgbaUnormSrgb, &level0, &mips).unwrap();
//! let tex = blob::load("tile.tex").unwrap(); // a Texture, chain and all
//! ```

use super::texture::{Texture, TextureFormat};
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"THREERSX";
const VERSION: u32 = 1;

fn format_code(f: TextureFormat) -> Option<u32> {
    Some(match f {
        TextureFormat::Bc1RgbaUnormSrgb => 1,
        TextureFormat::Bc7RgbaUnormSrgb => 2,
        TextureFormat::Rgba8UnormSrgb => 3,
        TextureFormat::Rgba8Unorm => 4,
        TextureFormat::R8Unorm => 5,
        // No code for Rgba16Float: HDR sources here are environment maps and
        // get built in memory, so writing one would be a mistake worth
        // reporting rather than a case worth supporting.
        TextureFormat::Rgba16Float => return None,
    })
}

fn format_from(code: u32) -> Option<TextureFormat> {
    Some(match code {
        1 => TextureFormat::Bc1RgbaUnormSrgb,
        2 => TextureFormat::Bc7RgbaUnormSrgb,
        3 => TextureFormat::Rgba8UnormSrgb,
        4 => TextureFormat::Rgba8Unorm,
        5 => TextureFormat::R8Unorm,
        _ => return None,
    })
}

/// Write level 0 and its chain.
///
/// Every level's length must match what `format.data_len` says for its size;
/// this refuses rather than write a file that will read back as noise.
pub fn save(
    path: impl AsRef<std::path::Path>,
    width: u32,
    height: u32,
    format: TextureFormat,
    level0: &[u8],
    mips: &[Vec<u8>],
) -> Result<(), String> {
    let code = format_code(format).ok_or_else(|| format!("{format:?} has no blob encoding"))?;
    let want = format.data_len(width, height);
    if level0.len() != want {
        return Err(format!(
            "level 0 is {} bytes, expected {want} for {width}x{height} {format:?}",
            level0.len()
        ));
    }
    let mut out =
        Vec::with_capacity(level0.len() + mips.iter().map(|m| m.len()).sum::<usize>() + 32);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&code.to_le_bytes());
    out.extend_from_slice(&(mips.len() as u32 + 1).to_le_bytes());
    out.extend_from_slice(level0);
    let (mut w, mut h) = (width, height);
    for (i, m) in mips.iter().enumerate() {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        let want = format.data_len(w, h);
        if m.len() != want {
            return Err(format!(
                "mip {} is {} bytes, expected {want} for {w}x{h}",
                i + 1,
                m.len()
            ));
        }
        out.extend_from_slice(m);
    }
    std::fs::write(path.as_ref(), out).map_err(|e| format!("{}: {e}", path.as_ref().display()))
}

/// Read one back as a `Texture`, chain attached and `flip_y` off.
///
/// `flip_y` is false because a block-compressed texture cannot be flipped by
/// reversing rows — the upload asserts on it — so the bytes are stored the way
/// they will be sampled.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<Texture, String> {
    let p = path.as_ref();
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if bytes.len() < 28 || &bytes[..8] != MAGIC {
        return Err(format!("{}: not a threers texture blob", p.display()));
    }
    let u32_at =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let version = u32_at(8);
    if version != VERSION {
        return Err(format!(
            "{}: version {version}, expected {VERSION}",
            p.display()
        ));
    }
    let (width, height) = (u32_at(12), u32_at(16));
    let format = format_from(u32_at(20))
        .ok_or_else(|| format!("{}: unknown format code {}", p.display(), u32_at(20)))?;
    let levels = u32_at(24) as usize;

    let mut off = 28;
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut chain: Vec<Arc<Vec<u8>>> = Vec::new();
    let mut level0 = Vec::new();
    for i in 0..levels {
        if i > 0 {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        let n = format.data_len(w, h);
        if off + n > bytes.len() {
            return Err(format!(
                "{}: truncated at level {i} — wanted {n} bytes for {w}x{h}, {} left",
                p.display(),
                bytes.len() - off
            ));
        }
        let data = bytes[off..off + n].to_vec();
        off += n;
        if i == 0 {
            level0 = data;
        } else {
            chain.push(Arc::new(data));
        }
    }
    let mut t = Texture::new(width, height, format, level0);
    t.flip_y = false;
    t.mips = chain;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::bc1;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("threers_blob_{name}"))
    }

    #[test]
    fn round_trips_a_chain() {
        let (w, h) = (64u32, 40u32);
        let px: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % 251) as u8, (i % 199) as u8, (i % 97) as u8, 255])
            .collect();
        let (l0, mips) = bc1::encode_with_mips(&px, w, h);
        let path = tmp("roundtrip.tex");
        save(&path, w, h, TextureFormat::Bc1RgbaUnormSrgb, &l0, &mips).unwrap();

        let t = load(&path).unwrap();
        assert_eq!((t.width, t.height), (w, h));
        assert_eq!(t.format, TextureFormat::Bc1RgbaUnormSrgb);
        assert_eq!(*t.data, l0);
        assert_eq!(t.mips.len(), mips.len());
        for (a, b) in t.mips.iter().zip(&mips) {
            assert_eq!(&**a.as_ref(), b.as_slice());
        }
        // Compressed data is stored the way it is sampled; flipping it is
        // impossible and the upload asserts on the attempt.
        assert!(!t.flip_y);
        std::fs::remove_file(&path).ok();
    }

    /// A stale or mismatched file must fail, not sample as noise.
    #[test]
    fn rejects_what_it_cannot_trust() {
        let path = tmp("bad.tex");

        std::fs::write(&path, b"not a texture at all").unwrap();
        assert!(load(&path)
            .unwrap_err()
            .contains("not a threers texture blob"));

        // Right header, wrong body: the classic stale-asset failure.
        let (w, h) = (32u32, 32u32);
        let px = vec![128u8; (w * h * 4) as usize];
        let (l0, mips) = bc1::encode_with_mips(&px, w, h);
        save(&path, w, h, TextureFormat::Bc1RgbaUnormSrgb, &l0, &mips).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 8);
        std::fs::write(&path, &bytes).unwrap();
        assert!(load(&path).unwrap_err().contains("truncated"));

        // And a level whose length disagrees with its size is refused on write.
        let err = save(
            &path,
            w,
            h,
            TextureFormat::Bc1RgbaUnormSrgb,
            &l0[..l0.len() - 8],
            &mips,
        )
        .unwrap_err();
        assert!(err.contains("level 0"), "{err}");
        std::fs::remove_file(&path).ok();
    }
}
