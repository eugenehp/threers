use crate::textures::{Texture, TextureFormat};

/// Radiance .hdr (RGBE) loader. Parses the ASCII header, then RLE-decompressed
/// RGBE scanlines into linear-RGB float values (stored here as Rgba8Unorm by
/// tonemap-clipping into 0–255). For a true HDR pipeline, swap to Rgba16Float
/// once the renderer exposes a half-float texture format.
pub struct HdrLoader;

#[derive(Debug)]
pub enum HdrError {
    BadMagic,
    BadHeader,
    BadScanline,
}

impl HdrLoader {
    /// Decode a Radiance `.hdr` to **linear f32 RGBA**, preserving the full
    /// dynamic range (a sun can legitimately be thousands of times brighter
    /// than the sky around it). Returns `(pixels, width, height)`.
    ///
    /// Use this for environment maps; [`Self::parse`] clips to 8-bit and is
    /// only appropriate for already-low-range images.
    pub fn parse_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32, u32), HdrError> {
        // Header is ASCII terminated by an empty line, then a dimension line
        // like "-Y H +X W", then binary RLE-RGBE scanlines.
        let mut pos = 0;
        if !bytes.starts_with(b"#?RADIANCE") && !bytes.starts_with(b"#?RGBE") {
            return Err(HdrError::BadMagic);
        }
        // Skip past first newline.
        while pos < bytes.len() && bytes[pos] != b'\n' {
            pos += 1;
        }
        pos += 1;
        // Read header lines until empty line.
        loop {
            let line_start = pos;
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            let line = &bytes[line_start..pos];
            pos += 1;
            if line.is_empty() {
                break;
            }
        }
        // Dimension line.
        let dim_start = pos;
        while pos < bytes.len() && bytes[pos] != b'\n' {
            pos += 1;
        }
        let dim_line =
            std::str::from_utf8(&bytes[dim_start..pos]).map_err(|_| HdrError::BadHeader)?;
        pos += 1;
        let mut height = 0usize;
        let mut width = 0usize;
        for tok in dim_line.split_whitespace() {
            if let Ok(n) = tok.parse::<usize>() {
                if height == 0 {
                    height = n;
                } else {
                    width = n;
                }
            }
        }
        if width == 0 || height == 0 {
            return Err(HdrError::BadHeader);
        }

        // Decode scanlines. Each scanline is `width` pixels of RGBE.
        let mut rgba: Vec<f32> = Vec::with_capacity(width * height * 4);
        for _ in 0..height {
            if pos + 4 > bytes.len() {
                return Err(HdrError::BadScanline);
            }
            let r = bytes[pos];
            let g = bytes[pos + 1];
            let b1 = bytes[pos + 2];
            let b2 = bytes[pos + 3];
            // Detect new-RLE scanline: `2,2,(width>>8),(width&0xff)` (width <= 32767).
            if r == 2 && g == 2 && b1 < 128 {
                let scan_w = ((b1 as usize) << 8) | b2 as usize;
                pos += 4;
                if scan_w != width {
                    return Err(HdrError::BadScanline);
                }
                let mut channels: [Vec<u8>; 4] = [
                    vec![0u8; width],
                    vec![0u8; width],
                    vec![0u8; width],
                    vec![0u8; width],
                ];
                for c in &mut channels {
                    let mut x = 0usize;
                    while x < width {
                        if pos >= bytes.len() {
                            return Err(HdrError::BadScanline);
                        }
                        let n = bytes[pos];
                        pos += 1;
                        if n > 128 {
                            // Run-length
                            let run = (n - 128) as usize;
                            if pos >= bytes.len() {
                                return Err(HdrError::BadScanline);
                            }
                            let val = bytes[pos];
                            pos += 1;
                            for _ in 0..run {
                                c[x] = val;
                                x += 1;
                            }
                        } else {
                            // Literal
                            let run = n as usize;
                            for _ in 0..run {
                                if pos >= bytes.len() {
                                    return Err(HdrError::BadScanline);
                                }
                                c[x] = bytes[pos];
                                pos += 1;
                                x += 1;
                            }
                        }
                    }
                }
                // Indexes all four channel planes at once; iterating one of
                // them would leave the other three indexed anyway.
                #[allow(clippy::needless_range_loop)]
                for x in 0..width {
                    let rgb = rgbe_to_linear(
                        channels[0][x],
                        channels[1][x],
                        channels[2][x],
                        channels[3][x],
                    );
                    rgba.extend_from_slice(&rgb);
                }
            } else {
                // Old-style flat RGBE scanline (no RLE).
                pos += 4;
                let rgb = rgbe_to_linear(r, g, b1, b2);
                rgba.extend_from_slice(&rgb);
                for _ in 1..width {
                    if pos + 4 > bytes.len() {
                        return Err(HdrError::BadScanline);
                    }
                    let rgb =
                        rgbe_to_linear(bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]);
                    rgba.extend_from_slice(&rgb);
                    pos += 4;
                }
            }
        }

        Ok((rgba, width as u32, height as u32))
    }

    /// Decode to an 8-bit texture, Reinhard-compressed rather than hard-clipped.
    ///
    /// Retained for callers that want a plain LDR image. Anything doing
    /// image-based lighting should use [`Self::parse_f32`] instead — clipping
    /// an HDR environment to 1.0 is what forces the "tiny sun" workaround,
    /// because a clipped sun cannot be brighter than the sky beside it.
    pub fn parse(bytes: &[u8]) -> Result<Texture, HdrError> {
        let (px, w, h) = Self::parse_f32(bytes)?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for c in px.chunks(4) {
            for i in 0..3 {
                let v = c.get(i).copied().unwrap_or(0.0).max(0.0);
                out.push(((v / (1.0 + v)).clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            out.push(255);
        }
        Ok(Texture::new(w, h, TextureFormat::Rgba8Unorm, out))
    }
}

/// Shared-exponent RGBE → linear f32 RGBA. No clamping: that is the whole point.
fn rgbe_to_linear(r: u8, g: u8, b: u8, e: u8) -> [f32; 4] {
    if e == 0 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let f = (2.0_f32).powi(e as i32 - 128 - 8);
    [r as f32 * f, g as f32 * f, b as f32 * f, 1.0]
}
