//! APNG (Animated PNG) encoder — a universally web-supported **transparent**
//! animation format (pure Rust, wasm-safe).
//!
//! APNG is the pragmatic cross-browser answer for transparent motion: every
//! major browser (Chrome, Firefox, Safari, Edge) renders it in an `<img>`, with
//! a full 8-bit alpha channel. For smaller files or video containers see the
//! sibling [`crate::gif`], [`crate::vp9`] / [`crate::webm`],
//! and [`crate::hevc`] modules.
//!
//! This writes truecolor-with-alpha (`color_type = 6`) frames. The DEFLATE stage
//! currently uses *stored* (uncompressed) blocks — valid and decodable
//! everywhere; real DEFLATE compression is a follow-up (the file is correct
//! either way, just larger).

/// CRC-32 (ISO 3309 / PNG) over a byte slice.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Adler-32 (zlib) over a byte slice.
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Wrap `data` in a zlib stream using DEFLATE *stored* (uncompressed) blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header: CM=8, CINFO=7, FLEVEL fastest
    if data.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]); // final empty block
    } else {
        let mut i = 0;
        while i < data.len() {
            let len = (data.len() - i).min(0xFFFF);
            let is_final = (i + len >= data.len()) as u8;
            out.push(is_final); // BFINAL | BTYPE(00 = stored)
            out.push((len & 0xFF) as u8);
            out.push((len >> 8) as u8);
            let nlen = !(len as u16);
            out.push((nlen & 0xFF) as u8);
            out.push((nlen >> 8) as u8);
            out.extend_from_slice(&data[i..i + len]);
            i += len;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Append a PNG chunk (`length | type | data | crc`) to `out`.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Filter+pack RGBA scanlines (filter type 0 = none) then zlib-compress.
fn frame_zlib(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = (width * 4) as usize;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0); // filter type: none
        raw.extend_from_slice(&rgba[row * stride..(row + 1) * stride]);
    }
    zlib_stored(&raw)
}

/// Builds an APNG from RGBA frames.
pub struct ApngEncoder {
    width: u32,
    height: u32,
    /// Loop count (`0` = infinite).
    plays: u32,
    frames: Vec<(Vec<u8>, u16, u16)>, // (rgba, delay_num, delay_den)
}

impl ApngEncoder {
    /// New encoder for `width × height` frames looping `plays` times (`0` = forever).
    pub fn new(width: u32, height: u32, plays: u32) -> Self {
        Self {
            width,
            height,
            plays,
            frames: Vec::new(),
        }
    }

    /// Add a frame: tightly-packed RGBA8 (`width × height × 4`), shown for
    /// `delay_num / delay_den` seconds.
    pub fn add_frame(&mut self, rgba: &[u8], delay_num: u16, delay_den: u16) {
        assert_eq!(
            rgba.len(),
            (self.width * self.height * 4) as usize,
            "rgba size"
        );
        self.frames.push((rgba.to_vec(), delay_num, delay_den));
    }

    /// Serialize the APNG byte stream.
    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]); // signature

        // IHDR: 8-bit truecolor + alpha (color type 6).
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&self.width.to_be_bytes());
        ihdr.extend_from_slice(&self.height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth, color type, compression, filter, interlace
        chunk(&mut out, b"IHDR", &ihdr);

        // acTL: animation control.
        let mut actl = Vec::new();
        actl.extend_from_slice(&(self.frames.len() as u32).to_be_bytes());
        actl.extend_from_slice(&self.plays.to_be_bytes());
        chunk(&mut out, b"acTL", &actl);

        let mut seq: u32 = 0;
        for (i, (rgba, dnum, dden)) in self.frames.iter().enumerate() {
            // fcTL for every frame.
            let mut fctl = Vec::new();
            fctl.extend_from_slice(&seq.to_be_bytes());
            seq += 1;
            fctl.extend_from_slice(&self.width.to_be_bytes());
            fctl.extend_from_slice(&self.height.to_be_bytes());
            fctl.extend_from_slice(&0u32.to_be_bytes()); // x_offset
            fctl.extend_from_slice(&0u32.to_be_bytes()); // y_offset
            fctl.extend_from_slice(&dnum.to_be_bytes());
            fctl.extend_from_slice(&dden.to_be_bytes());
            fctl.push(0); // dispose_op = NONE
            fctl.push(0); // blend_op = SOURCE (overwrite, incl. alpha)
            chunk(&mut out, b"fcTL", &fctl);

            let zdata = frame_zlib(self.width, self.height, rgba);
            if i == 0 {
                chunk(&mut out, b"IDAT", &zdata);
            } else {
                // fdAT: sequence number + frame data.
                let mut fdat = Vec::with_capacity(4 + zdata.len());
                fdat.extend_from_slice(&seq.to_be_bytes());
                seq += 1;
                fdat.extend_from_slice(&zdata);
                chunk(&mut out, b"fdAT", &fdat);
            }
        }
        chunk(&mut out, b"IEND", &[]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_and_adler_known_values() {
        // CRC32 of "IEND" (the empty-IEND chunk's crc is a PNG constant).
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn apng_structure() {
        let (w, h) = (4u32, 4u32);
        let mut enc = ApngEncoder::new(w, h, 0);
        let f0 = vec![255u8; (w * h * 4) as usize];
        let mut f1 = vec![0u8; (w * h * 4) as usize];
        for p in f1.chunks_mut(4) {
            p[3] = 128; // semi-transparent
        }
        enc.add_frame(&f0, 1, 10);
        enc.add_frame(&f1, 1, 10);
        let png = enc.finish();

        assert_eq!(&png[1..4], b"PNG");
        for tag in [b"IHDR", b"acTL", b"fcTL", b"IDAT", b"fdAT", b"IEND"] {
            assert!(
                png.windows(4).any(|w| w == tag),
                "missing chunk {:?}",
                std::str::from_utf8(tag)
            );
        }
    }
}
