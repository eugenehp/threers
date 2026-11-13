//! JPEG decoding to RGBA, the counterpart of [`png`](super::png) for the
//! images most packaged assets carry. A thin layer over `zune-jpeg` (pure
//! Rust, wasm-safe), behind the `jpeg` feature.

/// A decoded JPEG: 8-bit RGBA, rows top-first, alpha opaque.
#[derive(Debug, Clone)]
pub struct JpegImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Whether `bytes` start like a JPEG (an SOI marker, then another marker).
pub fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

/// Decode a baseline or progressive JPEG to RGBA.
///
/// Every layout comes out as four channels: a greyscale file is returned
/// with one channel whatever output was asked for, and is spread to grey
/// RGB here, so a caller can hand any result to a texture as RGBA.
pub fn decode_jpeg(bytes: &[u8]) -> Result<JpegImage, String> {
    use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let pixels = decoder.decode().map_err(|e| format!("jpeg: {e:?}"))?;
    let info = decoder.info().ok_or("jpeg: no header")?;
    let (width, height) = (info.width as usize, info.height as usize);
    let count = width * height;
    if count == 0 {
        return Err("jpeg: empty image".into());
    }
    let rgba = match pixels.len() / count {
        4 => pixels,
        3 => pixels.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        1 => pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        n => return Err(format!("jpeg: {n} bytes a pixel")),
    };
    if rgba.len() != count * 4 {
        return Err("jpeg: short image".into());
    }
    Ok(JpegImage { width: width as u32, height: height as u32, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 × 16, the top half red and the bottom half blue.
    const RGB: &[u8] = include_bytes!("../loaders/testdata/jpeg/rgb.jpg");
    /// 8 × 8, one channel, mid grey.
    const GREY: &[u8] = include_bytes!("../loaders/testdata/jpeg/grey.jpg");

    fn near(a: &[u8], b: [u8; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (*x as i32 - y as i32).abs() <= 12)
    }

    /// Rows come out top-first, in RGBA.
    #[test]
    fn a_colour_jpeg_decodes_to_rgba_top_row_first() {
        assert!(is_jpeg(RGB));
        let image = decode_jpeg(RGB).expect("decodes");
        assert_eq!((image.width, image.height, image.rgba.len()), (16, 16, 16 * 16 * 4));
        let px = |x: usize, y: usize| &image.rgba[(y * 16 + x) * 4..][..4];
        assert!(near(px(4, 2), [255, 0, 0, 255]), "top is red: {:?}", px(4, 2));
        assert!(near(px(4, 13), [0, 0, 255, 255]), "bottom is blue: {:?}", px(4, 13));
    }

    /// A greyscale file has one channel whatever was asked for; it still
    /// comes out as four, grey and opaque.
    #[test]
    fn a_greyscale_jpeg_comes_out_as_rgba_too() {
        let image = decode_jpeg(GREY).expect("decodes");
        assert_eq!((image.width, image.height, image.rgba.len()), (8, 8, 8 * 8 * 4));
        assert!(image.rgba.chunks_exact(4).all(|p| near(p, [128, 128, 128, 255])), "{:?}", &image.rgba[..4]);
    }

    /// Anything else is told no rather than misread.
    #[test]
    fn other_bytes_are_not_a_jpeg() {
        let png = crate::utils::png::encode_png(1, 1, &[1, 2, 3, 255]);
        assert!(!is_jpeg(&png));
        assert!(decode_jpeg(&png).is_err());
        assert!(decode_jpeg(&RGB[..RGB.len() / 3]).is_err(), "a truncated file is an error");
    }
}
