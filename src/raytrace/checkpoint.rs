//! Film checkpoint I/O and minimal EXR export for progressive renders.
//!
//! Checkpoints store raw accumulation state — not resolved means — so a long
//! render can resume without re-tracing samples already in the film.

use super::film::{Film, Pixel};

const MAGIC: &[u8; 4] = b"RTFC";
const VERSION: u32 = 1;

/// Raw film state suitable for save/load.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmCheckpoint {
    width: u32,
    height: u32,
    samples: u32,
    pixels: Vec<Pixel>,
}

impl FilmCheckpoint {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }

    pub fn from_film(film: &Film) -> Self {
        Self {
            width: film.width(),
            height: film.height(),
            samples: film.samples(),
            pixels: film.pixels().to_vec(),
        }
    }

    pub fn apply_to(&self, film: &mut Film) -> Result<(), String> {
        if film.width() != self.width || film.height() != self.height {
            return Err(format!(
                "film size {}×{} does not match checkpoint {}×{}",
                film.width(),
                film.height(),
                self.width,
                self.height
            ));
        }
        let expected = (self.width as usize) * (self.height as usize);
        if self.pixels.len() != expected {
            return Err(format!(
                "checkpoint has {} pixels, expected {expected}",
                self.pixels.len()
            ));
        }
        film.pixels_mut().copy_from_slice(&self.pixels);
        film.set_samples(self.samples);
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.pixels.len() * 56);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.samples.to_le_bytes());
        for p in &self.pixels {
            for c in p.color {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&p.alpha.to_le_bytes());
            for c in p.albedo {
                out.extend_from_slice(&c.to_le_bytes());
            }
            for c in p.normal {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&p.depth.to_le_bytes());
            out.extend_from_slice(&p.depth_samples.to_le_bytes());
            out.extend_from_slice(&p.samples.to_le_bytes());
            // On disk: legacy Σx² so old checkpoints remain readable.
            out.extend_from_slice(&super::film::sum_sq_from_m2(p).to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..4] != MAGIC {
            return Err("bad checkpoint magic".into());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("unsupported checkpoint version {version}"));
        }
        let width = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let samples = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        if width == 0 || height == 0 {
            return Err("invalid checkpoint dimensions".into());
        }
        let count = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "checkpoint dimensions overflow".to_string())?;
        let pixel_bytes = 56;
        let payload = 20 + count * pixel_bytes;
        if bytes.len() != payload {
            return Err(format!(
                "checkpoint size mismatch: got {} bytes, expected {payload}",
                bytes.len()
            ));
        }
        let mut pixels = Vec::with_capacity(count);
        let mut pos = 20;
        for _ in 0..count {
            let read_f32 = |p: &mut usize| -> Result<f32, String> {
                if *p + 4 > bytes.len() {
                    return Err("truncated checkpoint".into());
                }
                let v = f32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
                *p += 4;
                Ok(v)
            };
            let read_u32 = |p: &mut usize| -> Result<u32, String> {
                if *p + 4 > bytes.len() {
                    return Err("truncated checkpoint".into());
                }
                let v = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap());
                *p += 4;
                Ok(v)
            };
            let mut color = [0.0f32; 3];
            for c in &mut color {
                *c = read_f32(&mut pos)?;
            }
            let alpha = read_f32(&mut pos)?;
            let mut albedo = [0.0f32; 3];
            for c in &mut albedo {
                *c = read_f32(&mut pos)?;
            }
            let mut normal = [0.0f32; 3];
            for c in &mut normal {
                *c = read_f32(&mut pos)?;
            }
            let depth = read_f32(&mut pos)?;
            let depth_samples = read_u32(&mut pos)?;
            let pixel_samples = read_u32(&mut pos)?;
            let sum_sq = read_f32(&mut pos)?;
            pixels.push(Pixel {
                color,
                alpha,
                albedo,
                normal,
                depth,
                depth_samples,
                samples: pixel_samples,
                lum_sq: super::film::m2_from_sum_sq(color, pixel_samples, sum_sq),
            });
        }
        Ok(Self {
            width,
            height,
            samples,
            pixels,
        })
    }
}

const EXR_MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

/// Encode linear RGBA f32 (`width × height × 4`) as a ZIP-compressed scanline EXR.
pub fn encode_exr_rgba(width: u32, height: u32, hdr: &[f32]) -> Result<Vec<u8>, String> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Err("invalid EXR dimensions".into());
    }
    let expected = w * h * 4;
    if hdr.len() != expected {
        return Err(format!(
            "hdr slice has {} floats, expected {expected} for {width}×{height}",
            hdr.len()
        ));
    }

    let mut f = Vec::new();
    f.extend_from_slice(&EXR_MAGIC);
    f.extend_from_slice(&2u32.to_le_bytes());

    let attr = |name: &str, ty: &str, data: &[u8], f: &mut Vec<u8>| {
        f.extend_from_slice(name.as_bytes());
        f.push(0);
        f.extend_from_slice(ty.as_bytes());
        f.push(0);
        f.extend_from_slice(&(data.len() as u32).to_le_bytes());
        f.extend_from_slice(data);
    };

    // chlist: channels in alphabetical order, HALF samples.
    let mut chlist = Vec::new();
    for name in ["A", "B", "G", "R"] {
        chlist.extend_from_slice(name.as_bytes());
        chlist.push(0);
        chlist.extend_from_slice(&1i32.to_le_bytes()); // HALF
        chlist.extend_from_slice(&[0, 0, 0, 0]);
        chlist.extend_from_slice(&1i32.to_le_bytes());
        chlist.extend_from_slice(&1i32.to_le_bytes());
    }
    chlist.push(0);
    attr("channels", "chlist", &chlist, &mut f);
    attr("compression", "compression", &[3u8], &mut f); // ZIP
    let mut dw = Vec::new();
    for v in [0i32, 0, width as i32 - 1, height as i32 - 1] {
        dw.extend_from_slice(&v.to_le_bytes());
    }
    attr("dataWindow", "box2i", &dw, &mut f);
    f.push(0); // end of header

    let scanlines_per_block = 16;
    let chunks = h.div_ceil(scanlines_per_block);
    f.extend_from_slice(&vec![0u8; chunks * 8]);

    let bytes_per_scanline = w * 8; // four HALF channels
    for chunk in 0..chunks {
        let first_y = chunk * scanlines_per_block;
        let lines = scanlines_per_block.min(h - first_y);
        let mut raw = Vec::with_capacity(lines * bytes_per_scanline);
        for line in 0..lines {
            let y = first_y + line;
            let row = y * w * 4;
            for ch in 0..4usize {
                for x in 0..w {
                    let v = hdr[row + x * 4 + ch];
                    raw.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
        }
        let payload = zlib_stored(&exr_pack(&raw));
        f.extend_from_slice(&(first_y as i32).to_le_bytes());
        f.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        f.extend_from_slice(&payload);
    }
    Ok(f)
}

fn f32_to_f16(v: f32) -> u16 {
    crate::renderer::gpu_texture::f32_to_f16_bits(v)
}

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

fn zlib_stored(payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    for (i, chunk) in payload.chunks(0xffff).enumerate() {
        let last = (i + 1) * 0xffff >= payload.len();
        out.push(if last { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&[0, 0, 0, 0]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3;

    #[test]
    fn film_checkpoint_roundtrip() {
        let mut film = Film::new(3, 2);
        for (i, p) in film.pixels_mut().iter_mut().enumerate() {
            let t = i as f32 * 0.1;
            p.add(Vector3::new(t, t + 1.0, t + 2.0), 0.5 + t * 0.01);
            p.add_aux(
                Vector3::new(0.2, 0.3, 0.4),
                Vector3::new(0.0, 1.0, 0.0),
                1.5 + t,
            );
        }
        film.set_samples(4);

        let cp = FilmCheckpoint::from_film(&film);
        let bytes = cp.to_bytes();
        let back = FilmCheckpoint::from_bytes(&bytes).expect("decode");
        assert_eq!(cp.width(), back.width());
        assert_eq!(cp.height(), back.height());
        assert_eq!(cp.samples(), back.samples());

        let mut restored = Film::new(3, 2);
        back.apply_to(&mut restored).expect("apply");
        assert_eq!(restored.width(), film.width());
        assert_eq!(restored.height(), film.height());
        assert_eq!(restored.samples(), film.samples());
        for (a, b) in restored.pixels().iter().zip(film.pixels()) {
            assert_eq!(a.color, b.color);
            assert_eq!(a.alpha, b.alpha);
            assert_eq!(a.albedo, b.albedo);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.depth, b.depth);
            assert_eq!(a.depth_samples, b.depth_samples);
            assert_eq!(a.samples, b.samples);
            assert!(
                (a.lum_sq - b.lum_sq).abs() < 1e-3 * (1.0 + a.lum_sq.abs()),
                "M₂ roundtrip {} vs {}",
                a.lum_sq,
                b.lum_sq
            );
        }
    }

    #[test]
    fn film_checkpoint_rejects_bad_magic() {
        assert!(FilmCheckpoint::from_bytes(b"XXXX").is_err());
    }

    #[test]
    fn encode_exr_rgba_decodes() {
        use crate::loaders::ExrLoader;

        let (w, h) = (4u32, 8u32);
        let mut hdr = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                hdr[i] = x as f32 / w as f32;
                hdr[i + 1] = y as f32 / h as f32;
                hdr[i + 2] = 0.5;
                hdr[i + 3] = 1.0;
            }
        }
        let exr = encode_exr_rgba(w, h, &hdr).expect("encode");
        let tex = ExrLoader::parse_hdr(&exr).expect("decode");
        assert_eq!(tex.width, w);
        assert_eq!(tex.height, h);
        assert_eq!(tex.data.len(), (w * h * 8) as usize);
    }
}
