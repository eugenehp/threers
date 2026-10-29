//! Dependency-free GIF87a/GIF89a decoder.
//!
//! The compositing canvas starts as transparent black, and
//! [`DisposalMethod::RestoreBackground`] clears to transparent black. This is
//! intentional: it preserves transparent GIF round-trips instead of treating
//! the logical-screen background color as opaque.

use std::error::Error;
use std::fmt;

/// Why a GIF could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GifError {
    /// Byte stream ended before a complete block.
    Truncated,
    /// Missing or invalid `GIF87a` / `GIF89a` signature.
    BadSignature,
    /// Signature was present but not a supported version.
    BadVersion,
    /// LZW bitstream could not be decoded.
    BadLzw,
    /// Malformed extension or image block.
    BadBlock,
    /// Width/height or rectangle arithmetic overflowed.
    Overflow,
    /// Expected more bytes in a sub-block.
    UnexpectedEof,
    /// Required color table missing or oversized.
    InvalidPalette,
    /// Disposal method value is outside 0..=3.
    InvalidDisposal,
    /// Requested frame index exceeds [`GifInfo::frame_count`].
    FrameOutOfBounds,
}

impl fmt::Display for GifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Truncated => "truncated GIF",
            Self::BadSignature => "invalid GIF signature",
            Self::BadVersion => "unsupported GIF version",
            Self::BadLzw => "invalid GIF LZW stream",
            Self::BadBlock => "invalid GIF block",
            Self::Overflow => "GIF dimensions overflow",
            Self::UnexpectedEof => "unexpected end of GIF data",
            Self::InvalidPalette => "missing or invalid GIF color table",
            Self::InvalidDisposal => "invalid GIF disposal method",
            Self::FrameOutOfBounds => "GIF frame index out of bounds",
        };
        f.write_str(message)
    }
}

impl Error for GifError {}

/// GIF version from the six-byte signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GifVersion {
    /// Original GIF87a.
    Gif87a,
    /// GIF89a (extensions, transparency, Netscape loop, …).
    Gif89a,
}

/// How the decoder treats the canvas after displaying a frame (GCE field).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DisposalMethod {
    /// No disposal specified.
    #[default]
    None = 0,
    /// Leave the graphic in place.
    DoNotDispose = 1,
    /// Restore the frame rectangle to background (transparent black here).
    RestoreBackground = 2,
    /// Restore the canvas as it was before this frame.
    RestorePrevious = 3,
}

/// Logical-screen metadata from a GIF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GifInfo {
    /// Logical-screen width in pixels.
    pub width: u32,
    /// Logical-screen height in pixels.
    pub height: u32,
    /// `GIF87a` or `GIF89a`.
    pub version: GifVersion,
    /// The global colour table, if the file has one.
    pub global_palette: Option<Vec<[u8; 3]>>,
    /// Index into the global table of the background colour.
    pub background_index: u8,
    /// The Netscape loop field as stored. `Some(0)` means infinite looping;
    /// `None` means no Netscape loop extension was present.
    pub loop_count: Option<u16>,
    /// Image descriptors in the file.
    pub frame_count: usize,
    /// Comment extension payloads, in file order.
    pub comments: Vec<Vec<u8>>,
}

/// Per-frame descriptor metadata (before LZW decode / compositing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GifFrameMeta {
    /// Frame delay in centiseconds.
    pub delay_cs: u16,
    /// What happens to this frame's rectangle before the next one is drawn.
    pub disposal: DisposalMethod,
    /// The palette index drawn as transparent, if any.
    pub transparent: Option<u8>,
    /// Left edge of the frame's rectangle on the canvas.
    pub x: u16,
    /// Top edge of the frame's rectangle on the canvas.
    pub y: u16,
    /// Width of the frame's rectangle.
    pub w: u16,
    /// Height of the frame's rectangle.
    pub h: u16,
    /// Whether rows are stored in the four-pass interlaced order.
    pub interlaced: bool,
    /// The frame's own colour table, overriding the global one.
    pub local_palette: Option<Vec<[u8; 3]>>,
}

/// One composited full-canvas frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GifFrame {
    /// Full logical-screen RGBA canvas after this frame has been composited.
    pub rgba: Vec<u8>,
    /// Delay numerator (centiseconds from the GCE).
    pub delay_num: u16,
    /// Delay denominator (always `100` for GIF centiseconds).
    pub delay_den: u16,
    /// The frame's descriptor, as parsed.
    pub meta: GifFrameMeta,
}

/// The old name of [`GifFrame`].
#[deprecated(
    since = "0.0.7",
    note = "renamed `GifFrame`, so it is not mistaken for the video decoder's `codec::decode::DecodedFrame`"
)]
pub type DecodedFrame = GifFrame;

#[derive(Clone, Debug)]
struct IndexedFrame {
    meta: GifFrameMeta,
    min_code_size: u8,
    compressed: Vec<u8>,
}

/// Indexed GIF reader with lazy or eager frame decoding.
///
/// Call [`Self::open`] to parse the container, then [`Self::decode_frame`] /
/// [`Self::decode_all`]. Random access to frame `i` decodes `0..=i` in order so
/// disposal state stays correct.
pub struct GifDecoder {
    info: GifInfo,
    frames: Vec<IndexedFrame>,
    decoded: Vec<GifFrame>,
    canvas: Vec<u8>,
    restore_previous: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Default)]
struct GraphicControl {
    delay_cs: u16,
    disposal: DisposalMethod,
    transparent: Option<u8>,
}

impl GifDecoder {
    /// Parses the header and indexes image data. A missing final trailer is
    /// accepted when all preceding blocks are complete.
    pub fn open(bytes: &[u8]) -> Result<Self, GifError> {
        if bytes.len() < 6 {
            return Err(GifError::Truncated);
        }
        if &bytes[..3] != b"GIF" {
            return Err(GifError::BadSignature);
        }
        let version = match &bytes[3..6] {
            b"87a" => GifVersion::Gif87a,
            b"89a" => GifVersion::Gif89a,
            _ => return Err(GifError::BadVersion),
        };

        let mut reader = Reader::new(bytes, 6);
        let width = reader.u16()? as u32;
        let height = reader.u16()? as u32;
        if width == 0 || height == 0 {
            return Err(GifError::BadBlock);
        }
        let packed = reader.byte()?;
        let background_index = reader.byte()?;
        let _pixel_aspect = reader.byte()?;
        let global_palette = if packed & 0x80 != 0 {
            Some(reader.palette(table_len(packed))?)
        } else {
            None
        };

        let mut frames = Vec::new();
        let mut comments = Vec::new();
        let mut loop_count = None;
        let mut gce = GraphicControl::default();

        while !reader.at_end() {
            match reader.byte()? {
                0x3b => break,
                0x2c => {
                    let x = reader.u16()?;
                    let y = reader.u16()?;
                    let w = reader.u16()?;
                    let h = reader.u16()?;
                    if w == 0 || h == 0 {
                        return Err(GifError::BadBlock);
                    }
                    let right = u32::from(x)
                        .checked_add(u32::from(w))
                        .ok_or(GifError::Overflow)?;
                    let bottom = u32::from(y)
                        .checked_add(u32::from(h))
                        .ok_or(GifError::Overflow)?;
                    if right > width || bottom > height {
                        return Err(GifError::Overflow);
                    }
                    let image_packed = reader.byte()?;
                    let local_palette = if image_packed & 0x80 != 0 {
                        Some(reader.palette(table_len(image_packed))?)
                    } else {
                        None
                    };
                    if local_palette.is_none() && global_palette.is_none() {
                        return Err(GifError::InvalidPalette);
                    }
                    let min_code_size = reader.byte()?;
                    if !(2..=8).contains(&min_code_size) {
                        return Err(GifError::BadLzw);
                    }
                    let compressed = reader.sub_blocks()?;
                    frames.push(IndexedFrame {
                        meta: GifFrameMeta {
                            delay_cs: gce.delay_cs,
                            disposal: gce.disposal,
                            transparent: gce.transparent,
                            x,
                            y,
                            w,
                            h,
                            interlaced: image_packed & 0x40 != 0,
                            local_palette,
                        },
                        min_code_size,
                        compressed,
                    });
                    gce = GraphicControl::default();
                }
                0x21 => {
                    let label = reader.byte()?;
                    match label {
                        0xf9 => {
                            if reader.byte()? != 4 {
                                return Err(GifError::BadBlock);
                            }
                            let control = reader.byte()?;
                            let disposal = match (control >> 2) & 7 {
                                0 => DisposalMethod::None,
                                1 => DisposalMethod::DoNotDispose,
                                2 => DisposalMethod::RestoreBackground,
                                3 => DisposalMethod::RestorePrevious,
                                _ => return Err(GifError::InvalidDisposal),
                            };
                            let delay_cs = reader.u16()?;
                            let transparent_index = reader.byte()?;
                            if reader.byte()? != 0 {
                                return Err(GifError::BadBlock);
                            }
                            gce = GraphicControl {
                                delay_cs,
                                disposal,
                                transparent: (control & 1 != 0).then_some(transparent_index),
                            };
                        }
                        0xfe => comments.push(reader.sub_blocks()?),
                        0xff => {
                            let app_len = reader.byte()? as usize;
                            let app = reader.take(app_len)?;
                            let data = reader.sub_blocks()?;
                            if app == b"NETSCAPE2.0" && data.len() >= 3 && data[0] == 1 {
                                loop_count = Some(u16::from_le_bytes([data[1], data[2]]));
                            }
                        }
                        // Plain-text and unrecognized extensions are chains of
                        // data sub-blocks and can safely be ignored.
                        _ => {
                            reader.sub_blocks()?;
                        }
                    }
                }
                _ => return Err(GifError::BadBlock),
            }
        }

        let pixel_bytes = canvas_len(width, height)?;
        let frame_count = frames.len();
        Ok(Self {
            info: GifInfo {
                width,
                height,
                version,
                global_palette,
                background_index,
                loop_count,
                frame_count,
                comments,
            },
            frames,
            decoded: Vec::new(),
            canvas: Vec::with_capacity(pixel_bytes.min(4096)),
            restore_previous: None,
        })
    }

    /// The file-level information parsed by [`open`](Self::open).
    pub fn info(&self) -> &GifInfo {
        &self.info
    }

    /// Frames in the file.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Frame `i`'s descriptor, without decoding it.
    pub fn frame_meta(&self, i: usize) -> Result<&GifFrameMeta, GifError> {
        self.frames
            .get(i)
            .map(|frame| &frame.meta)
            .ok_or(GifError::FrameOutOfBounds)
    }

    /// Decodes sequentially through `i`; already decoded frames are cached.
    pub fn decode_frame(&mut self, i: usize) -> Result<GifFrame, GifError> {
        if i >= self.frames.len() {
            return Err(GifError::FrameOutOfBounds);
        }
        while self.decoded.len() <= i {
            self.decode_next()?;
        }
        Ok(self.decoded[i].clone())
    }

    /// Decode and composite every frame.
    pub fn decode_all(mut self) -> Result<Vec<GifFrame>, GifError> {
        while self.decoded.len() < self.frames.len() {
            self.decode_next()?;
        }
        Ok(self.decoded)
    }

    fn decode_next(&mut self) -> Result<(), GifError> {
        let index = self.decoded.len();
        if index == 0 {
            let len = canvas_len(self.info.width, self.info.height)?;
            self.canvas
                .try_reserve_exact(len)
                .map_err(|_| GifError::Overflow)?;
            self.canvas.resize(len, 0);
        } else {
            let previous = &self.frames[index - 1].meta;
            match previous.disposal {
                DisposalMethod::None | DisposalMethod::DoNotDispose => {}
                DisposalMethod::RestoreBackground => {
                    clear_rect(&mut self.canvas, self.info.width, previous)
                }
                DisposalMethod::RestorePrevious => {
                    let backup = self.restore_previous.take().ok_or(GifError::BadBlock)?;
                    self.canvas = backup;
                }
            }
        }

        let frame = &self.frames[index];
        self.restore_previous = if frame.meta.disposal == DisposalMethod::RestorePrevious {
            Some(self.canvas.clone())
        } else {
            None
        };
        let expected = usize::from(frame.meta.w)
            .checked_mul(usize::from(frame.meta.h))
            .ok_or(GifError::Overflow)?;
        let mut indices = lzw_decode(&frame.compressed, frame.min_code_size, expected)?;
        if frame.meta.interlaced {
            indices = deinterlace(&indices, frame.meta.w as usize, frame.meta.h as usize);
        }
        let palette = frame
            .meta
            .local_palette
            .as_deref()
            .or(self.info.global_palette.as_deref())
            .ok_or(GifError::InvalidPalette)?;
        composite(
            &mut self.canvas,
            self.info.width,
            &frame.meta,
            palette,
            &indices,
        )?;
        self.decoded.push(GifFrame {
            rgba: self.canvas.clone(),
            delay_num: frame.meta.delay_cs,
            delay_den: 100,
            meta: frame.meta.clone(),
        });
        Ok(())
    }
}

/// Decode an entire GIF into logical-screen metadata and composited RGBA frames.
pub fn decode_gif(bytes: &[u8]) -> Result<(GifInfo, Vec<GifFrame>), GifError> {
    let decoder = GifDecoder::open(bytes)?;
    let info = decoder.info.clone();
    let frames = decoder.decode_all()?;
    Ok((info, frames))
}

fn table_len(packed: u8) -> usize {
    1usize << (usize::from(packed & 7) + 1)
}

fn canvas_len(width: u32, height: u32) -> Result<usize, GifError> {
    usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(GifError::Overflow)
}

fn clear_rect(canvas: &mut [u8], canvas_width: u32, meta: &GifFrameMeta) {
    let stride = canvas_width as usize * 4;
    for y in usize::from(meta.y)..usize::from(meta.y) + usize::from(meta.h) {
        let start = y * stride + usize::from(meta.x) * 4;
        let end = start + usize::from(meta.w) * 4;
        canvas[start..end].fill(0);
    }
}

fn composite(
    canvas: &mut [u8],
    canvas_width: u32,
    meta: &GifFrameMeta,
    palette: &[[u8; 3]],
    indices: &[u8],
) -> Result<(), GifError> {
    let w = usize::from(meta.w);
    let stride = canvas_width as usize;
    for (offset, &color_index) in indices.iter().enumerate() {
        if meta.transparent == Some(color_index) {
            continue;
        }
        let rgb = palette
            .get(color_index as usize)
            .ok_or(GifError::InvalidPalette)?;
        let x = usize::from(meta.x) + offset % w;
        let y = usize::from(meta.y) + offset / w;
        let dst = (y * stride + x) * 4;
        canvas[dst..dst + 3].copy_from_slice(rgb);
        canvas[dst + 3] = 255;
    }
    Ok(())
}

fn deinterlace(input: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0; input.len()];
    let mut source = 0;
    for (start, step) in [(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
        for y in (start..height).step_by(step) {
            output[y * width..(y + 1) * width].copy_from_slice(&input[source..source + width]);
            source += width;
        }
    }
    output
}

fn lzw_decode(data: &[u8], min_code_size: u8, expected: usize) -> Result<Vec<u8>, GifError> {
    let clear = 1u16 << min_code_size;
    let end = clear + 1;
    let mut dictionary = Vec::<Vec<u8>>::with_capacity(4096);
    reset_dictionary(&mut dictionary, clear);
    let mut next_code = end + 1;
    let mut code_size = u32::from(min_code_size) + 1;
    let mut bits = BitReader::new(data);
    let mut previous: Option<Vec<u8>> = None;
    let mut output = Vec::with_capacity(expected);
    let mut saw_end = false;

    while let Some(code) = bits.read(code_size) {
        if code == clear {
            reset_dictionary(&mut dictionary, clear);
            next_code = end + 1;
            code_size = u32::from(min_code_size) + 1;
            previous = None;
            continue;
        }
        if code == end {
            saw_end = true;
            break;
        }
        if code >= 4096 {
            return Err(GifError::BadLzw);
        }

        let entry = if let Some(entry) = dictionary.get(code as usize) {
            entry.clone()
        } else if code == next_code {
            let mut entry = previous.clone().ok_or(GifError::BadLzw)?;
            let first = *entry.first().ok_or(GifError::BadLzw)?;
            entry.push(first);
            entry
        } else {
            return Err(GifError::BadLzw);
        };
        if output
            .len()
            .checked_add(entry.len())
            .ok_or(GifError::Overflow)?
            > expected
        {
            return Err(GifError::BadLzw);
        }
        output.extend_from_slice(&entry);

        // Once the required pixels are complete, the encoder writes EOI
        // without adding another dictionary entry. Avoid a spurious width
        // bump at exactly that boundary.
        if output.len() < expected {
            if let Some(previous_entry) = previous.as_ref() {
                if next_code < 4096 {
                    let mut new_entry = previous_entry.clone();
                    new_entry.push(entry[0]);
                    if dictionary.len() != next_code as usize {
                        return Err(GifError::BadLzw);
                    }
                    dictionary.push(new_entry);
                    next_code += 1;
                    // The decoder reconstructs an encoder dictionary entry one
                    // emitted code later. Grow as soon as the reconstructed
                    // `next_code` reaches the encoder's width boundary, before
                    // reading the following code.
                    if next_code == (1 << code_size) && code_size < 12 {
                        code_size += 1;
                    }
                }
            }
        }
        previous = Some(entry);
    }

    if !saw_end || output.len() != expected {
        return Err(GifError::BadLzw);
    }
    Ok(output)
}

fn reset_dictionary(dictionary: &mut Vec<Vec<u8>>, clear: u16) {
    dictionary.clear();
    dictionary.extend((0..clear).map(|value| vec![value as u8]));
    // Reserve slots for Clear and EOI so code indices line up directly.
    dictionary.push(Vec::new());
    dictionary.push(Vec::new());
}

struct BitReader<'a> {
    data: &'a [u8],
    byte: usize,
    buffer: u32,
    bits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte: 0,
            buffer: 0,
            bits: 0,
        }
    }

    fn read(&mut self, width: u32) -> Option<u16> {
        while self.bits < width {
            let byte = *self.data.get(self.byte)?;
            self.byte += 1;
            self.buffer |= u32::from(byte) << self.bits;
            self.bits += 8;
        }
        let code = self.buffer & ((1 << width) - 1);
        self.buffer >>= width;
        self.bits -= width;
        Some(code as u16)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], position: usize) -> Self {
        Self { bytes, position }
    }

    fn at_end(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn byte(&mut self) -> Result<u8, GifError> {
        let byte = self
            .bytes
            .get(self.position)
            .copied()
            .ok_or(GifError::UnexpectedEof)?;
        self.position += 1;
        Ok(byte)
    }

    fn u16(&mut self) -> Result<u16, GifError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], GifError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(GifError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or(GifError::UnexpectedEof)?;
        self.position = end;
        Ok(bytes)
    }

    fn palette(&mut self, length: usize) -> Result<Vec<[u8; 3]>, GifError> {
        let bytes = self.take(length.checked_mul(3).ok_or(GifError::Overflow)?)?;
        Ok(bytes
            .chunks_exact(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect())
    }

    fn sub_blocks(&mut self) -> Result<Vec<u8>, GifError> {
        let mut output = Vec::new();
        loop {
            let length = self.byte()? as usize;
            if length == 0 {
                return Ok(output);
            }
            let block = self.take(length)?;
            output
                .try_reserve(block.len())
                .map_err(|_| GifError::Overflow)?;
            output.extend_from_slice(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gif::{DisposalMode, GifEncoder, GifOptions, PaletteMode};

    fn solid(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        color.repeat((w * h) as usize)
    }

    fn close(actual: &[u8], expected: &[u8], tolerance: i16) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((i16::from(*a) - i16::from(*b)).abs() <= tolerance);
        }
    }

    #[test]
    fn round_trip_solid_frames() {
        let mut encoder = GifEncoder::new(4, 3, 0)
            .dither(false)
            .diff_rects(false)
            .palette_mode(PaletteMode::Local);
        encoder.add_frame(&solid(4, 3, [240, 20, 10, 255]), 3, 100);
        encoder.add_frame(&solid(4, 3, [10, 30, 230, 255]), 7, 100);
        let (info, frames) = decode_gif(&encoder.finish()).unwrap();
        assert_eq!((info.width, info.height, info.frame_count), (4, 3, 2));
        assert_eq!(info.loop_count, Some(0));
        close(&frames[0].rgba[..4], &[240, 20, 10, 255], 4);
        close(&frames[1].rgba[..4], &[10, 30, 230, 255], 4);
        assert_eq!((frames[1].delay_num, frames[1].delay_den), (7, 100));
    }

    #[test]
    fn encoder_lzw_decodes_every_index() {
        let mut pixels = Vec::new();
        for i in 0..256u16 {
            pixels.extend_from_slice(&[(i & 255) as u8, (255 - i) as u8, 17, 255]);
        }
        let mut encoder = GifEncoder::new(16, 16, 1)
            .colors(256)
            .dither(false)
            .diff_rects(false)
            .transparency(false);
        encoder.add_frame(&pixels, 1, 100);
        let (_, frames) = decode_gif(&encoder.finish()).unwrap();
        assert_eq!(frames[0].rgba.len(), pixels.len());
        assert!(frames[0].rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn truncated_and_bad_signature_are_errors() {
        assert!(matches!(
            GifDecoder::open(b"GIF89"),
            Err(GifError::Truncated)
        ));
        assert!(matches!(
            GifDecoder::open(b"NOPE89a"),
            Err(GifError::BadSignature)
        ));
    }

    #[test]
    fn interlaced_content_matches_plain_content() {
        let mut pixels = Vec::new();
        for y in 0..13u8 {
            for x in 0..9u8 {
                pixels.extend_from_slice(&[x * 20, y * 17, x ^ y, 255]);
            }
        }
        let options = GifOptions::default()
            .colors(256)
            .dither(false)
            .diff_rects(false)
            .transparency(false);
        let mut plain = GifEncoder::new(9, 13, 1).options(options.clone());
        plain.add_frame(&pixels, 1, 100);
        let mut interlaced = GifEncoder::new(9, 13, 1).options(options.interlace(true));
        interlaced.add_frame(&pixels, 1, 100);
        let (_, plain_frames) = decode_gif(&plain.finish()).unwrap();
        let (_, interlaced_frames) = decode_gif(&interlaced.finish()).unwrap();
        assert_eq!(plain_frames[0].rgba, interlaced_frames[0].rgba);
        assert!(interlaced_frames[0].meta.interlaced);
    }

    #[test]
    fn dirty_rect_keeps_unchanged_canvas_pixels() {
        let first = solid(8, 8, [10, 20, 30, 255]);
        let mut second = first.clone();
        for y in 3..5 {
            for x in 2..4 {
                let offset = (y * 8 + x) * 4;
                second[offset..offset + 4].copy_from_slice(&[240, 5, 10, 255]);
            }
        }
        let mut encoder = GifEncoder::new(8, 8, 1)
            .dither(false)
            .diff_rects(true)
            .disposal(DisposalMode::Keep);
        encoder.add_frame(&first, 1, 100);
        encoder.add_frame(&second, 1, 100);
        let bytes = encoder.finish();
        let mut decoder = GifDecoder::open(&bytes).unwrap();
        decoder.decode_frame(0).unwrap();
        let second_frame = decoder.decode_frame(1).unwrap();
        close(&second_frame.rgba[0..4], &[10, 20, 30, 255], 4);
        let changed = (3 * 8 + 2) * 4;
        close(
            &second_frame.rgba[changed..changed + 4],
            &[240, 5, 10, 255],
            4,
        );
    }
}
