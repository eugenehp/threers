//! GIF89a animated GIF encoder — pure Rust, wasm-safe.
//!
//! Companion to [`crate::codec::apng`]: same RGBA frame API, with production
//! knobs for palette size, transparency, dithering, dirty-rect differencing,
//! local/global palettes, octree quantization, disposal, lossy differencing,
//! LZW clear policies, comments, and interlacing.
//!
//! [`GifWriter`] writes local-palette GIFs incrementally to any [`std::io::Write`].
//! Decoding is provided by [`decode`] (`GifDecoder` / [`decode_gif`]).
//!
//! # Streaming
//!
//! [`GifEncoder::encode_iter`] consumes frames as they arrive. With
//! [`PaletteMode::Local`] (default for streaming) only the previous canvas is
//! retained — no full-sequence buffer. [`PaletteMode::Global`] still collects
//! frames once to build a shared palette.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "native-codec")] {
//! use threers::codec::gif::{GifEncoder, PaletteMode};
//! let mut enc = GifEncoder::new(2, 2, 1)
//!     .colors(4)
//!     .dither(false)
//!     .diff_rects(false)
//!     .palette_mode(PaletteMode::Global)
//!     .transparency(false);
//! enc.add_frame(&[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255], 1, 10);
//! let gif = enc.finish();
//! assert_eq!(&gif[..6], b"GIF89a");
//! # }
//! ```

pub mod decode;

pub use decode::{
    decode_gif, DecodedFrame, DisposalMethod, GifDecoder, GifError, GifFrameMeta, GifInfo,
    GifVersion,
};

use std::collections::HashMap;
use std::io::{self, Write};

/// How a frame's pixels are disposed before the following frame.
///
/// Matches the GIF89a Graphic Control Extension disposal field. [`Self::Auto`]
/// picks Keep / Background / Previous from the composited canvas to shrink
/// subsequent dirty rectangles while remaining decoder-compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisposalMode {
    /// Choose Keep, Background, or Previous based on the next frame.
    #[default]
    Auto,
    /// Disposal 0 — no action required after the frame.
    None,
    /// Disposal 1 — leave pixels in place for the next frame.
    Keep,
    /// Disposal 2 — restore the frame rectangle to background (transparent).
    Background,
    /// Disposal 3 — restore the canvas as it was before this frame was drawn.
    Previous,
}

/// Policy used to clear the GIF LZW dictionary during compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LzwClearMode {
    /// Clear as soon as the table fills (4096 entries). Broadest compatibility.
    #[default]
    WhenFull,
    /// Freeze inserts at capacity; clear when the match rate drops.
    Deferred,
    /// May clear before the table fills when compression ratio worsens.
    Adaptive,
}

/// How the color table(s) are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PaletteMode {
    /// One global palette from all frames (best size when colors are shared).
    Global,
    /// Per-frame local color table (better quality on scene changes).
    #[default]
    Local,
    /// Global when it fits; otherwise fall back to local for that frame.
    Auto,
}

/// Quantization algorithm for reducing opaque colors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuantizerKind {
    /// Gervautz–Purgathofer octree (default — good quality / speed).
    #[default]
    Octree,
    /// Classic median-cut.
    MedianCut,
}

/// Tunables for [`GifEncoder`] / [`GifWriter`] / [`encode_gif`].
#[derive(Clone, Debug)]
pub struct GifOptions {
    /// Max palette entries including the transparent slot (`2..=256`).
    pub max_colors: u16,
    /// Map low-alpha pixels to a reserved transparent index.
    pub transparency: bool,
    /// `alpha < threshold` → transparent when [`Self::transparency`] is on.
    pub alpha_threshold: u8,
    /// Floyd–Steinberg dithering after quantization.
    pub dither: bool,
    /// Encode only changed bounding boxes; unchanged pixels → transparent.
    pub diff_rects: bool,
    /// Global, per-frame local, or automatic palette selection.
    pub palette_mode: PaletteMode,
    /// Octree (default) or median-cut color reduction.
    pub quantizer: QuantizerKind,
    /// Frame disposal policy (see [`DisposalMode`]).
    pub disposal: DisposalMode,
    /// Loss tolerance (`0..=100`) for dirty-pixel detection and index coherence.
    /// `0` is lossless relative to the other options; higher values shrink files.
    pub lossy: u8,
    /// When to emit LZW clear codes (see [`LzwClearMode`]).
    pub lzw_clear: LzwClearMode,
    /// Write frames with the GIF interlacing flag and 4-pass row order.
    pub interlace: bool,
    /// Comment extension payloads written after the header.
    pub comments: Vec<Vec<u8>>,
}

impl Default for GifOptions {
    fn default() -> Self {
        Self {
            max_colors: 256,
            transparency: true,
            alpha_threshold: 128,
            dither: true,
            diff_rects: true,
            palette_mode: PaletteMode::Local,
            quantizer: QuantizerKind::Octree,
            disposal: DisposalMode::Auto,
            lossy: 0,
            lzw_clear: LzwClearMode::WhenFull,
            interlace: false,
            comments: Vec::new(),
        }
    }
}

impl GifOptions {
    /// Clamp palette size to `2..=256`.
    pub fn colors(mut self, n: u16) -> Self {
        assert!(
            (2..=256).contains(&n),
            "GIF palette size must be 2..=256 (got {n})"
        );
        self.max_colors = n;
        self
    }
    /// Enable or disable the transparent palette index.
    pub fn transparency(mut self, on: bool) -> Self {
        self.transparency = on;
        self
    }
    /// Alpha cutoff for treating a pixel as transparent.
    pub fn alpha_threshold(mut self, t: u8) -> Self {
        self.alpha_threshold = t;
        self
    }
    /// Enable or disable Floyd–Steinberg dithering.
    pub fn dither(mut self, on: bool) -> Self {
        self.dither = on;
        self
    }
    /// Enable or disable dirty-rectangle frame differencing.
    pub fn diff_rects(mut self, on: bool) -> Self {
        self.diff_rects = on;
        self
    }
    /// Select global / local / auto palettes.
    pub fn palette_mode(mut self, mode: PaletteMode) -> Self {
        self.palette_mode = mode;
        self
    }
    /// Select the quantizer used when colors exceed the palette budget.
    pub fn quantizer(mut self, q: QuantizerKind) -> Self {
        self.quantizer = q;
        self
    }
    /// Override automatic disposal selection.
    pub fn disposal(mut self, mode: DisposalMode) -> Self {
        self.disposal = mode;
        self
    }
    /// Lossiness `0..=100` (`0` preserves exact indexing decisions).
    pub fn lossy(mut self, loss: u8) -> Self {
        assert!(loss <= 100, "GIF lossy setting must be 0..=100");
        self.lossy = loss;
        self
    }
    /// LZW dictionary clear policy.
    pub fn lzw_clear(mut self, mode: LzwClearMode) -> Self {
        self.lzw_clear = mode;
        self
    }
    /// Emit interlaced image descriptors.
    pub fn interlace(mut self, on: bool) -> Self {
        self.interlace = on;
        self
    }
    /// Append a GIF comment extension (may be called multiple times).
    pub fn comment(mut self, bytes: impl AsRef<[u8]>) -> Self {
        self.comments.push(bytes.as_ref().to_vec());
        self
    }
}

struct Frame {
    rgba: Vec<u8>,
    delay_num: u16,
    delay_den: u16,
}

/// Builds an animated GIF from RGBA frames.
///
/// Buffer frames with [`Self::add_frame`] / [`Self::add_frame_owned`], then
/// [`Self::finish`], or stream with [`Self::encode_iter`]. For writing directly
/// to a file or socket, prefer [`GifWriter`].
pub struct GifEncoder {
    width: u32,
    height: u32,
    plays: u32,
    opts: GifOptions,
    frames: Vec<Frame>,
}

/// Incremental GIF writer using per-frame local palettes.
///
/// Writes the header immediately, then one image block per [`Self::write_frame`].
/// Does **not** support [`PaletteMode::Global`] (returns `InvalidInput`);
/// [`PaletteMode::Auto`] is treated as local.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "native-codec")] {
/// use threers::codec::gif::{GifOptions, GifWriter, PaletteMode};
/// let mut out = Vec::new();
/// let opts = GifOptions::default()
///     .colors(8)
///     .dither(false)
///     .diff_rects(false)
///     .transparency(false)
///     .palette_mode(PaletteMode::Local);
/// let mut w = GifWriter::new(&mut out, 2, 2, 1, opts).unwrap();
/// w.write_frame(&[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255], 1, 10).unwrap();
/// w.finish().unwrap();
/// assert_eq!(&out[..6], b"GIF89a");
/// # }
/// ```
pub struct GifWriter<W: Write> {
    out: W,
    width: usize,
    height: usize,
    opts: GifOptions,
    canvas: Vec<u8>,
    pending: Option<Frame>,
    frames_written: usize,
}

impl<W: Write> GifWriter<W> {
    /// Create a writer and emit the GIF header (+ optional Netscape loop / comments).
    pub fn new(
        mut out: W,
        width: u32,
        height: u32,
        plays: u32,
        mut opts: GifOptions,
    ) -> io::Result<Self> {
        if width == 0 || height == 0 || width > u16::MAX as u32 || height > u16::MAX as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid GIF dimensions",
            ));
        }
        if opts.palette_mode == PaletteMode::Global {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GifWriter does not support global palettes",
            ));
        }
        opts.palette_mode = PaletteMode::Local;
        write_header(&mut out, width, height, plays, None, 0)?;
        write_comments(&mut out, &opts.comments)?;
        Ok(Self {
            out,
            width: width as usize,
            height: height as usize,
            canvas: vec![0; width as usize * height as usize * 4],
            opts,
            pending: None,
            frames_written: 0,
        })
    }

    /// Queue one RGBA frame. Delay is `delay_num / delay_den` seconds
    /// (converted to GIF centiseconds). The previous frame may be flushed now
    /// so automatic disposal can inspect this frame.
    pub fn write_frame(&mut self, rgba: &[u8], delay_num: u16, delay_den: u16) -> io::Result<()> {
        if rgba.len() != self.canvas.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "rgba size"));
        }
        if self.pending.is_some() {
            self.flush_pending(Some(rgba))?;
        }
        self.pending = Some(Frame {
            rgba: rgba.to_vec(),
            delay_num,
            delay_den: delay_den.max(1),
        });
        Ok(())
    }

    /// Flush any pending frame, then write a Comment Extension block.
    pub fn write_comment(&mut self, text: &[u8]) -> io::Result<()> {
        self.flush_pending(None)?;
        write_comment(&mut self.out, text)
    }

    fn flush_pending(&mut self, next: Option<&[u8]>) -> io::Result<()> {
        let Some(frame) = self.pending.take() else {
            return Ok(());
        };
        let previous = (self.frames_written != 0).then_some(self.canvas.as_slice());
        let (rect, indexed, palette, transparent, _) = prepare_frame(
            self.width,
            self.height,
            &frame.rgba,
            previous,
            &self.opts,
            true,
        );
        let disposal = choose_disposal(
            self.width,
            self.height,
            &self.canvas,
            &frame.rgba,
            next,
            rect,
            transparent.is_some(),
            &self.opts,
        );
        write_image(
            &mut self.out,
            delay_to_cs(frame.delay_num, frame.delay_den),
            disposal,
            transparent,
            rect,
            &indexed,
            &palette,
            true,
            &self.opts,
        )?;
        advance_canvas(
            self.width,
            self.height,
            &mut self.canvas,
            &frame.rgba,
            rect,
            disposal,
            &self.opts,
        );
        self.frames_written += 1;
        Ok(())
    }

    /// Flush the last pending frame, write the trailer (`0x3B`), and return the
    /// underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.flush_pending(None)?;
        self.out.write_all(&[0x3B])?;
        Ok(self.out)
    }
}

impl GifEncoder {
    /// New encoder; defaults enable dithering, dirty-rects, and local palettes.
    pub fn new(width: u32, height: u32, plays: u32) -> Self {
        assert!(width > 0 && height > 0 && width <= 0xFFFF && height <= 0xFFFF);
        Self {
            width,
            height,
            plays,
            opts: GifOptions::default(),
            frames: Vec::new(),
        }
    }

    pub fn options(mut self, opts: GifOptions) -> Self {
        self.opts = opts;
        self
    }

    pub fn colors(mut self, n: u16) -> Self {
        self.opts = self.opts.colors(n);
        self
    }
    pub fn transparency(mut self, on: bool) -> Self {
        self.opts = self.opts.transparency(on);
        self
    }
    pub fn alpha_threshold(mut self, t: u8) -> Self {
        self.opts = self.opts.alpha_threshold(t);
        self
    }
    pub fn dither(mut self, on: bool) -> Self {
        self.opts = self.opts.dither(on);
        self
    }
    pub fn diff_rects(mut self, on: bool) -> Self {
        self.opts = self.opts.diff_rects(on);
        self
    }
    pub fn palette_mode(mut self, mode: PaletteMode) -> Self {
        self.opts = self.opts.palette_mode(mode);
        self
    }
    pub fn quantizer(mut self, q: QuantizerKind) -> Self {
        self.opts = self.opts.quantizer(q);
        self
    }
    pub fn disposal(mut self, mode: DisposalMode) -> Self {
        self.opts = self.opts.disposal(mode);
        self
    }
    pub fn lossy(mut self, loss: u8) -> Self {
        self.opts = self.opts.lossy(loss);
        self
    }
    pub fn lzw_clear(mut self, mode: LzwClearMode) -> Self {
        self.opts = self.opts.lzw_clear(mode);
        self
    }
    pub fn interlace(mut self, on: bool) -> Self {
        self.opts = self.opts.interlace(on);
        self
    }
    pub fn comment(mut self, bytes: impl AsRef<[u8]>) -> Self {
        self.opts = self.opts.comment(bytes);
        self
    }

    /// Add a frame by moving RGBA bytes (no extra clone of the pixel buffer).
    pub fn add_frame_owned(&mut self, rgba: Vec<u8>, delay_num: u16, delay_den: u16) {
        assert_eq!(
            rgba.len(),
            (self.width * self.height * 4) as usize,
            "rgba size"
        );
        self.frames.push(Frame {
            rgba,
            delay_num,
            delay_den: delay_den.max(1),
        });
    }

    /// Add a frame from a borrowed RGBA buffer (one copy into the encoder).
    pub fn add_frame(&mut self, rgba: &[u8], delay_num: u16, delay_den: u16) {
        self.add_frame_owned(rgba.to_vec(), delay_num, delay_den);
    }

    /// Serialize all buffered frames.
    pub fn finish(self) -> Vec<u8> {
        assert!(!self.frames.is_empty(), "GIF needs at least one frame");
        let frames = self
            .frames
            .into_iter()
            .map(|f| (f.rgba, f.delay_num, f.delay_den));
        encode_sequence(self.width, self.height, self.plays, &self.opts, frames)
    }

    /// Encode an iterator of `(rgba, delay_num, delay_den)` frames.
    ///
    /// With [`PaletteMode::Local`] this streams (keeps only the previous canvas).
    /// With [`PaletteMode::Global`] / [`PaletteMode::Auto`] it may buffer frames
    /// to build a shared palette.
    pub fn encode_iter<I>(self, frames: I) -> Vec<u8>
    where
        I: IntoIterator<Item = (Vec<u8>, u16, u16)>,
    {
        encode_sequence(self.width, self.height, self.plays, &self.opts, frames)
    }
}

/// Encode RGBA frames to a GIF byte stream (same options as [`GifEncoder`]).
pub fn encode_gif<I>(width: u32, height: u32, plays: u32, opts: &GifOptions, frames: I) -> Vec<u8>
where
    I: IntoIterator<Item = (Vec<u8>, u16, u16)>,
{
    assert!(width > 0 && height > 0 && width <= 0xFFFF && height <= 0xFFFF);
    encode_sequence(width, height, plays, opts, frames)
}

fn encode_sequence<I>(width: u32, height: u32, plays: u32, opts: &GifOptions, frames: I) -> Vec<u8>
where
    I: IntoIterator<Item = (Vec<u8>, u16, u16)>,
{
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 4;

    match opts.palette_mode {
        PaletteMode::Local => encode_streaming(width, height, plays, opts, frames, expected),
        PaletteMode::Global | PaletteMode::Auto => {
            let collected: Vec<_> = frames
                .into_iter()
                .map(|(rgba, n, d)| {
                    assert_eq!(rgba.len(), expected, "rgba size");
                    (rgba, n, d.max(1))
                })
                .collect();
            assert!(!collected.is_empty(), "GIF needs at least one frame");
            if opts.palette_mode == PaletteMode::Global {
                encode_with_global(width, height, plays, opts, &collected)
            } else {
                // Auto: try global; if any frame's opaque unique colors exceed
                // budget badly, still use global for shared look — local only
                // when global palette is a poor fit for a frame (re-encode that
                // frame with LCT). Simplified: use global if total unique ≤ budget,
                // else stream local.
                let opaque_unique = count_opaque_unique(&collected, opts);
                if opaque_unique <= opts.max_colors as usize {
                    encode_with_global(width, height, plays, opts, &collected)
                } else {
                    encode_streaming(width, height, plays, opts, collected, expected)
                }
            }
        }
    }
}

fn count_opaque_unique(frames: &[(Vec<u8>, u16, u16)], opts: &GifOptions) -> usize {
    let mut set = HashMap::<[u8; 3], ()>::new();
    for (rgba, _, _) in frames {
        for px in rgba.chunks_exact(4) {
            if !(opts.transparency && px[3] < opts.alpha_threshold) {
                set.insert([px[0], px[1], px[2]], ());
            }
        }
    }
    set.len()
}

fn encode_streaming<I>(
    width: u32,
    height: u32,
    plays: u32,
    opts: &GifOptions,
    frames: I,
    expected: usize,
) -> Vec<u8>
where
    I: IntoIterator<Item = (Vec<u8>, u16, u16)>,
{
    let w = width as usize;
    let h = height as usize;
    let mut out = Vec::new();
    write_header(&mut out, width, height, plays, /*gct*/ None, 0).unwrap();
    write_comments(&mut out, &opts.comments).unwrap();
    let mut iter = frames.into_iter();
    let mut pending = iter.next().map(|(r, n, d)| (r, n, d.max(1)));
    assert!(pending.is_some(), "GIF needs at least one frame");
    let mut canvas = vec![0u8; expected];
    let mut frame_no = 0usize;
    while let Some((rgba, dnum, dden)) = pending.take() {
        assert_eq!(rgba.len(), expected, "rgba size");
        pending = iter.next().map(|(r, n, d)| (r, n, d.max(1)));
        if let Some((next, _, _)) = &pending {
            assert_eq!(next.len(), expected, "rgba size");
        }
        let (rect, indexed, palette, transparent, _) = prepare_frame(
            w,
            h,
            &rgba,
            (frame_no != 0).then_some(canvas.as_slice()),
            opts,
            true,
        );
        let disposal = choose_disposal(
            w,
            h,
            &canvas,
            &rgba,
            pending.as_ref().map(|f| f.0.as_slice()),
            rect,
            transparent.is_some(),
            opts,
        );
        write_image(
            &mut out,
            delay_to_cs(dnum, dden),
            disposal,
            transparent,
            rect,
            &indexed,
            &palette,
            true,
            opts,
        )
        .unwrap();
        advance_canvas(w, h, &mut canvas, &rgba, rect, disposal, opts);
        frame_no += 1;
    }
    out.push(0x3B);
    out
}

fn encode_with_global(
    width: u32,
    height: u32,
    plays: u32,
    opts: &GifOptions,
    frames: &[(Vec<u8>, u16, u16)],
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let (global_pal, global_tr) = build_palette_from_frames(frames, opts);
    let bg = global_tr.unwrap_or(0);
    let mut out = Vec::new();
    write_header(&mut out, width, height, plays, Some(&global_pal), bg).unwrap();
    write_comments(&mut out, &opts.comments).unwrap();
    let mut canvas = vec![0u8; w * h * 4];

    for (frame_no, (rgba, dnum, dden)) in frames.iter().enumerate() {
        let delay = delay_to_cs(*dnum, *dden);
        let (rect, indexed, _pal, transparent, _) = prepare_frame_with_palette(
            w,
            h,
            rgba,
            (frame_no != 0).then_some(canvas.as_slice()),
            opts,
            &global_pal,
            global_tr,
            false,
        );
        let disposal = choose_disposal(
            w,
            h,
            &canvas,
            rgba,
            frames.get(frame_no + 1).map(|f| f.0.as_slice()),
            rect,
            transparent.or(global_tr).is_some(),
            opts,
        );
        write_image(
            &mut out,
            delay,
            disposal,
            transparent.or(global_tr),
            rect,
            &indexed,
            &global_pal,
            false,
            opts,
        )
        .unwrap();
        advance_canvas(w, h, &mut canvas, rgba, rect, disposal, opts);
    }
    out.push(0x3B);
    out
}

#[derive(Clone, Copy)]
struct Rect {
    x: u16,
    y: u16,
    w: u16,
    h: u16,
}

fn full_rect(w: usize, h: usize) -> Rect {
    Rect {
        x: 0,
        y: 0,
        w: w as u16,
        h: h as u16,
    }
}

#[allow(clippy::too_many_arguments)]
fn choose_disposal(
    w: usize,
    h: usize,
    canvas: &[u8],
    rgba: &[u8],
    next: Option<&[u8]>,
    rect: Rect,
    has_transparency: bool,
    opts: &GifOptions,
) -> u8 {
    match opts.disposal {
        DisposalMode::None => return 0,
        DisposalMode::Keep => return 1,
        DisposalMode::Background => return 2,
        DisposalMode::Previous => return 3,
        DisposalMode::Auto => {}
    }

    let initially_empty = canvas.chunks_exact(4).all(|p| p[3] == 0);
    if initially_empty && has_transparency {
        return 2;
    }
    let Some(next) = next else {
        return if initially_empty {
            0
        } else if has_transparency {
            2
        } else {
            1
        };
    };

    let mut displayed = canvas.to_vec();
    compose_source(&mut displayed, rgba, opts);
    let mut background = displayed.clone();
    clear_rect(w, &mut background, rect);
    let previous = canvas;

    let score = |candidate: &[u8]| -> usize {
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut changed = false;
        for (pixel, (desired, under)) in next
            .chunks_exact(4)
            .zip(candidate.chunks_exact(4))
            .enumerate()
        {
            let desired_tr = opts.transparency && desired[3] < opts.alpha_threshold;
            let under_tr = opts.transparency && under[3] < opts.alpha_threshold;
            // A transparent source pixel cannot erase an opaque canvas pixel.
            if desired_tr && !under_tr {
                return usize::MAX;
            }
            if pixels_differ(desired, under, opts) {
                let x = pixel % w;
                let y = pixel / w;
                changed = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
        if changed {
            (max_x - min_x + 1) * (max_y - min_y + 1)
        } else {
            0
        }
    };
    let candidates = [
        (1u8, score(&displayed)),
        (2, score(&background)),
        (3, score(previous)),
    ];
    let best = candidates
        .into_iter()
        .min_by_key(|(mode, area)| {
            (
                *area,
                match mode {
                    1 => 0,
                    2 => 1,
                    _ => 2,
                },
            )
        })
        .unwrap()
        .0;
    if initially_empty && best == 1 {
        0
    } else {
        best
    }
}

fn compose_source(canvas: &mut [u8], rgba: &[u8], opts: &GifOptions) {
    for (dst, src) in canvas.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
        if !(opts.transparency && src[3] < opts.alpha_threshold)
            && (dst[3] == 0 || pixels_differ(dst, src, opts))
        {
            dst.copy_from_slice(src);
            dst[3] = 255;
        }
    }
}

fn clear_rect(stride_w: usize, canvas: &mut [u8], rect: Rect) {
    for y in rect.y as usize..rect.y as usize + rect.h as usize {
        for x in rect.x as usize..rect.x as usize + rect.w as usize {
            canvas[(y * stride_w + x) * 4..(y * stride_w + x + 1) * 4].fill(0);
        }
    }
}

fn advance_canvas(
    w: usize,
    _h: usize,
    canvas: &mut [u8],
    rgba: &[u8],
    rect: Rect,
    disposal: u8,
    opts: &GifOptions,
) {
    let previous = (disposal == 3).then(|| canvas.to_vec());
    compose_source(canvas, rgba, opts);
    if disposal == 2 {
        clear_rect(w, canvas, rect);
    } else if let Some(previous) = previous {
        canvas.copy_from_slice(&previous);
    }
}

/// Returns (rect, indices row-major in rect, palette, transparent, disposal).
fn prepare_frame(
    w: usize,
    h: usize,
    rgba: &[u8],
    prev: Option<&[u8]>,
    opts: &GifOptions,
    local: bool,
) -> (Rect, Vec<u8>, Vec<[u8; 3]>, Option<u8>, u8) {
    let (palette, transparent) = if local {
        build_palette_region(rgba, w, h, full_rect(w, h), opts)
    } else {
        unreachable!()
    };
    prepare_frame_with_palette(w, h, rgba, prev, opts, &palette, transparent, local)
}

#[allow(clippy::too_many_arguments)]
fn prepare_frame_with_palette(
    w: usize,
    h: usize,
    rgba: &[u8],
    prev: Option<&[u8]>,
    opts: &GifOptions,
    palette: &[[u8; 3]],
    transparent: Option<u8>,
    _local: bool,
) -> (Rect, Vec<u8>, Vec<[u8; 3]>, Option<u8>, u8) {
    let first = prev.is_none();
    let mut tr = transparent;

    // Dirty rect against previous canvas.
    let rect = if opts.diff_rects && !first {
        dirty_rect(w, h, rgba, prev.unwrap(), opts).unwrap_or_else(|| full_rect(w, h))
    } else {
        full_rect(w, h)
    };

    // Rebuild local palette from the dirty region when local + diff.
    let palette_owned: Vec<[u8; 3]>;
    let pal: &[[u8; 3]] = if _local && opts.diff_rects && !first {
        let (p, t) = build_palette_region(rgba, w, h, rect, opts);
        tr = t.or(tr);
        palette_owned = p;
        &palette_owned
    } else {
        palette_owned = palette.to_vec();
        &palette_owned
    };

    // Ensure we have a transparent index when using diff-rects (unchanged → skip).
    let need_diff_tr = opts.diff_rects && !first;
    let tr = if need_diff_tr && tr.is_none() {
        // Steal last slot / append if room.
        let mut p = palette_owned.clone();
        let idx = if p.len() < opts.max_colors as usize {
            let i = p.len();
            p.push([0, 0, 0]);
            i as u8
        } else {
            (p.len() - 1) as u8
        };
        let indexed = index_region(w, h, rgba, prev, rect, &p, Some(idx), opts);
        let disposal = 1; // do not dispose
        return (rect, indexed, p, Some(idx), disposal);
    } else {
        tr
    };

    let indexed = index_region(w, h, rgba, prev, rect, pal, tr, opts);

    // Disposal: first transparent frame → restore bg; diff frames → keep;
    // full opaque replace → none.
    let disposal = if first && tr.is_some() {
        2
    } else if opts.diff_rects && !first {
        1
    } else if tr.is_some() {
        2
    } else {
        0
    };

    (rect, indexed, pal.to_vec(), tr, disposal)
}

fn dirty_rect(w: usize, h: usize, cur: &[u8], prev: &[u8], opts: &GifOptions) -> Option<Rect> {
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            if pixels_differ(&cur[i..i + 4], &prev[i..i + 4], opts) {
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if !any {
        // Zero-delay empty change: encode 1×1 transparent.
        return Some(Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        });
    }
    Some(Rect {
        x: min_x as u16,
        y: min_y as u16,
        w: (max_x - min_x + 1) as u16,
        h: (max_y - min_y + 1) as u16,
    })
}

fn pixels_differ(a: &[u8], b: &[u8], opts: &GifOptions) -> bool {
    let a_tr = opts.transparency && a[3] < opts.alpha_threshold;
    let b_tr = opts.transparency && b[3] < opts.alpha_threshold;
    if a_tr != b_tr {
        return true;
    }
    if a_tr {
        return false;
    }
    if opts.lossy == 0 {
        return a[0] != b[0] || a[1] != b[1] || a[2] != b[2];
    }
    dist2([a[0], a[1], a[2]], [b[0], b[1], b[2]]) > lossy_dist2(opts.lossy)
}

#[allow(clippy::too_many_arguments)]
fn index_region(
    stride_w: usize,
    _stride_h: usize,
    rgba: &[u8],
    prev: Option<&[u8]>,
    rect: Rect,
    palette: &[[u8; 3]],
    transparent: Option<u8>,
    opts: &GifOptions,
) -> Vec<u8> {
    let rw = rect.w as usize;
    let rh = rect.h as usize;
    let mut out = vec![0u8; rw * rh];

    if opts.dither {
        // Working buffer of RGB(A) errors as f32 in the rect.
        let mut work = vec![0f32; rw * rh * 4];
        for ly in 0..rh {
            for lx in 0..rw {
                let x = rect.x as usize + lx;
                let y = rect.y as usize + ly;
                let i = (y * stride_w + x) * 4;
                let o = (ly * rw + lx) * 4;
                work[o] = rgba[i] as f32;
                work[o + 1] = rgba[i + 1] as f32;
                work[o + 2] = rgba[i + 2] as f32;
                work[o + 3] = rgba[i + 3] as f32;
            }
        }
        for ly in 0..rh {
            for lx in 0..rw {
                let x = rect.x as usize + lx;
                let y = rect.y as usize + ly;
                let i = (y * stride_w + x) * 4;
                let o = (ly * rw + lx) * 4;
                let pi = ly * rw + lx;

                // Unchanged vs previous → transparent (diff mode).
                if let (Some(prev), Some(ti)) = (prev, transparent) {
                    if opts.diff_rects && !pixels_differ(&rgba[i..i + 4], &prev[i..i + 4], opts) {
                        out[pi] = ti;
                        continue;
                    }
                }
                if let Some(t) = transparent {
                    if opts.transparency && work[o + 3] < opts.alpha_threshold as f32 {
                        out[pi] = t;
                        continue;
                    }
                }

                let rgb = [
                    work[o].clamp(0.0, 255.0).round() as u8,
                    work[o + 1].clamp(0.0, 255.0).round() as u8,
                    work[o + 2].clamp(0.0, 255.0).round() as u8,
                ];
                let neighbor = if lx > 0 {
                    Some(out[pi - 1])
                } else if ly > 0 {
                    Some(out[pi - rw])
                } else {
                    None
                };
                let idx = nearest_index_lossy(rgb, palette, transparent, neighbor, opts.lossy);
                out[pi] = idx;
                let q = palette[idx as usize];
                let er = work[o] - q[0] as f32;
                let eg = work[o + 1] - q[1] as f32;
                let eb = work[o + 2] - q[2] as f32;
                // Floyd–Steinberg
                let disperse = |work: &mut [f32], lx2: i32, ly2: i32, fr: f32| {
                    if lx2 < 0 || ly2 < 0 || lx2 >= rw as i32 || ly2 >= rh as i32 {
                        return;
                    }
                    let j = ((ly2 as usize) * rw + lx2 as usize) * 4;
                    work[j] += er * fr;
                    work[j + 1] += eg * fr;
                    work[j + 2] += eb * fr;
                };
                disperse(&mut work, lx as i32 + 1, ly as i32, 7.0 / 16.0);
                disperse(&mut work, lx as i32 - 1, ly as i32 + 1, 3.0 / 16.0);
                disperse(&mut work, lx as i32, ly as i32 + 1, 5.0 / 16.0);
                disperse(&mut work, lx as i32 + 1, ly as i32 + 1, 1.0 / 16.0);
            }
        }
    } else {
        for ly in 0..rh {
            for lx in 0..rw {
                let x = rect.x as usize + lx;
                let y = rect.y as usize + ly;
                let i = (y * stride_w + x) * 4;
                let pi = ly * rw + lx;
                if let (Some(prev), Some(ti)) = (prev, transparent) {
                    if opts.diff_rects && !pixels_differ(&rgba[i..i + 4], &prev[i..i + 4], opts) {
                        out[pi] = ti;
                        continue;
                    }
                }
                if let Some(t) =
                    transparent.filter(|_| opts.transparency && rgba[i + 3] < opts.alpha_threshold)
                {
                    out[pi] = t;
                } else {
                    let neighbor = if lx > 0 {
                        Some(out[pi - 1])
                    } else if ly > 0 {
                        Some(out[pi - rw])
                    } else {
                        None
                    };
                    out[pi] = nearest_index_lossy(
                        [rgba[i], rgba[i + 1], rgba[i + 2]],
                        palette,
                        transparent,
                        neighbor,
                        opts.lossy,
                    );
                }
            }
        }
    }
    out
}

fn write_header(
    out: &mut impl Write,
    width: u32,
    height: u32,
    plays: u32,
    gct: Option<&[[u8; 3]]>,
    bg: u8,
) -> io::Result<()> {
    out.write_all(b"GIF89a")?;
    out.write_all(&(width as u16).to_le_bytes())?;
    out.write_all(&(height as u16).to_le_bytes())?;
    if let Some(pal) = gct {
        let bits = palette_bits(pal.len());
        let size = 1usize << bits;
        out.write_all(&[0x80 | (((bits - 1) as u8) << 4) | ((bits - 1) as u8), bg, 0])?;
        for i in 0..size {
            if i < pal.len() {
                out.write_all(&pal[i])?;
            } else {
                out.write_all(&[0, 0, 0])?;
            }
        }
    } else {
        // No global table — size bits 0, flag clear.
        out.write_all(&[0, 0, 0])?;
    }

    if plays != 1 {
        out.write_all(&[0x21, 0xFF, 0x0B])?;
        out.write_all(b"NETSCAPE2.0")?;
        out.write_all(&[0x03, 0x01])?;
        let loops = if plays == 0 {
            0u16
        } else {
            plays.saturating_sub(1).min(u16::MAX as u32) as u16
        };
        out.write_all(&loops.to_le_bytes())?;
        out.write_all(&[0x00])?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_image(
    out: &mut impl Write,
    delay_cs: u16,
    disposal: u8,
    transparent: Option<u8>,
    rect: Rect,
    indices: &[u8],
    palette: &[[u8; 3]],
    local_table: bool,
    opts: &GifOptions,
) -> io::Result<()> {
    out.write_all(&[0x21, 0xF9, 0x04])?;
    let mut packed = (disposal & 7) << 2;
    if transparent.is_some() {
        packed |= 0x01;
    }
    out.write_all(&[packed])?;
    out.write_all(&delay_cs.to_le_bytes())?;
    out.write_all(&[transparent.unwrap_or(0), 0x00])?;

    out.write_all(&[0x2C])?;
    out.write_all(&rect.x.to_le_bytes())?;
    out.write_all(&rect.y.to_le_bytes())?;
    out.write_all(&rect.w.to_le_bytes())?;
    out.write_all(&rect.h.to_le_bytes())?;

    let bits = palette_bits(palette.len().max(2));
    let size = 1usize << bits;
    if local_table {
        out.write_all(&[0x80 | if opts.interlace { 0x40 } else { 0 } | ((bits - 1) as u8)])?;
        for i in 0..size {
            if i < palette.len() {
                out.write_all(&palette[i])?;
            } else {
                out.write_all(&[0, 0, 0])?;
            }
        }
    } else {
        out.write_all(&[if opts.interlace { 0x40 } else { 0 }])?;
    }

    let min_code_size = bits.max(2) as u8;
    let reordered;
    let indices = if opts.interlace {
        reordered = interlace_indices(indices, rect.w as usize, rect.h as usize);
        &reordered
    } else {
        indices
    };
    let lzw = lzw_encode(indices, min_code_size, opts.lzw_clear);
    out.write_all(&[min_code_size])?;
    write_sub_blocks(out, &lzw)?;
    Ok(())
}

fn interlace_indices(indices: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len());
    for (start, step) in [(0usize, 8usize), (4, 8), (2, 4), (1, 2)] {
        for y in (start..h).step_by(step) {
            out.extend_from_slice(&indices[y * w..(y + 1) * w]);
        }
    }
    out
}

fn write_comment(out: &mut impl Write, text: &[u8]) -> io::Result<()> {
    out.write_all(&[0x21, 0xFE])?;
    write_sub_blocks(out, text)
}

fn write_comments(out: &mut impl Write, comments: &[Vec<u8>]) -> io::Result<()> {
    for comment in comments {
        write_comment(out, comment)?;
    }
    Ok(())
}

fn delay_to_cs(num: u16, den: u16) -> u16 {
    let cs = (u32::from(num) * 100) / u32::from(den.max(1));
    cs.min(u16::MAX as u32) as u16
}

fn palette_bits(n: usize) -> usize {
    let mut b = 1;
    while (1 << b) < n.max(2) {
        b += 1;
    }
    b.clamp(1, 8)
}

fn build_palette_from_frames(
    frames: &[(Vec<u8>, u16, u16)],
    opts: &GifOptions,
) -> (Vec<[u8; 3]>, Option<u8>) {
    let mut samples = Vec::new();
    let mut needs_tr = false;
    for (rgba, _, _) in frames {
        for px in rgba.chunks_exact(4) {
            if opts.transparency && px[3] < opts.alpha_threshold {
                needs_tr = true;
            } else {
                samples.push([px[0], px[1], px[2]]);
            }
        }
    }
    finish_palette(samples, needs_tr, opts)
}

fn build_palette_region(
    rgba: &[u8],
    w: usize,
    _h: usize,
    rect: Rect,
    opts: &GifOptions,
) -> (Vec<[u8; 3]>, Option<u8>) {
    let mut samples = Vec::new();
    let mut needs_tr = false;
    for ly in 0..rect.h as usize {
        for lx in 0..rect.w as usize {
            let x = rect.x as usize + lx;
            let y = rect.y as usize + ly;
            let i = (y * w + x) * 4;
            if opts.transparency && rgba[i + 3] < opts.alpha_threshold {
                needs_tr = true;
            } else {
                samples.push([rgba[i], rgba[i + 1], rgba[i + 2]]);
            }
        }
    }
    finish_palette(samples, needs_tr, opts)
}

fn finish_palette(
    samples: Vec<[u8; 3]>,
    needs_tr: bool,
    opts: &GifOptions,
) -> (Vec<[u8; 3]>, Option<u8>) {
    let max_colors = opts.max_colors as usize;
    let max_opaque = if needs_tr {
        max_colors.saturating_sub(1).max(1)
    } else {
        max_colors
    };

    // Fast path: unique colors already fit.
    let mut counts: HashMap<[u8; 3], u32> = HashMap::new();
    for c in &samples {
        *counts.entry(*c).or_insert(0) += 1;
    }
    let mut palette = if counts.len() <= max_opaque {
        let mut v: Vec<_> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.into_iter().map(|(c, _)| c).collect()
    } else {
        match opts.quantizer {
            QuantizerKind::Octree => octree_quantize(&samples, max_opaque),
            QuantizerKind::MedianCut => {
                let mut v: Vec<_> = counts.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                median_cut(&v, max_opaque)
            }
        }
    };

    if palette.is_empty() {
        palette.push([0, 0, 0]);
    }
    if palette.len() > max_opaque {
        palette.truncate(max_opaque);
    }

    let transparent = if needs_tr {
        let idx = palette.len();
        if idx < max_colors {
            palette.push([0, 0, 0]);
            Some(idx as u8)
        } else {
            Some((palette.len() - 1) as u8)
        }
    } else {
        None
    };
    (palette, transparent)
}

// ---- Octree quantizer -------------------------------------------------------

struct OctNode {
    children: [Option<Box<OctNode>>; 8],
    leaf: bool,
    count: u32,
    r: u64,
    g: u64,
    b: u64,
}

impl OctNode {
    fn new_leaf() -> Self {
        Self {
            children: Default::default(),
            leaf: true,
            count: 0,
            r: 0,
            g: 0,
            b: 0,
        }
    }
    fn new_branch() -> Self {
        Self {
            children: Default::default(),
            leaf: false,
            count: 0,
            r: 0,
            g: 0,
            b: 0,
        }
    }
}

fn octree_quantize(samples: &[[u8; 3]], max_colors: usize) -> Vec<[u8; 3]> {
    let mut root = OctNode::new_branch();
    let mut leaf_count = 0usize;
    for c in samples {
        insert_color(&mut root, *c, 0, &mut leaf_count);
        while leaf_count > max_colors {
            if !reduce_tree(&mut root, &mut leaf_count) {
                break;
            }
        }
    }
    let mut out = Vec::new();
    collect_leaves(&root, &mut out);
    if out.is_empty() {
        out.push([0, 0, 0]);
    }
    if out.len() > max_colors {
        out.truncate(max_colors);
    }
    out
}

fn child_index(rgb: [u8; 3], level: u8) -> usize {
    let shift = 7 - level;
    (((rgb[0] >> shift) & 1) << 2 | ((rgb[1] >> shift) & 1) << 1 | ((rgb[2] >> shift) & 1)) as usize
}

fn insert_color(node: &mut OctNode, rgb: [u8; 3], level: u8, leaf_count: &mut usize) {
    if node.leaf || level == 8 {
        if !node.leaf {
            node.leaf = true;
            *leaf_count += 1;
        }
        node.count += 1;
        node.r += rgb[0] as u64;
        node.g += rgb[1] as u64;
        node.b += rgb[2] as u64;
        return;
    }
    let i = child_index(rgb, level);
    if node.children[i].is_none() {
        if level == 7 {
            node.children[i] = Some(Box::new(OctNode::new_leaf()));
            *leaf_count += 1;
        } else {
            node.children[i] = Some(Box::new(OctNode::new_branch()));
        }
    }
    insert_color(
        node.children[i].as_mut().unwrap(),
        rgb,
        level + 1,
        leaf_count,
    );
}

fn reduce_tree(node: &mut OctNode, leaf_count: &mut usize) -> bool {
    // Find a deepest foldable branch (has only leaf children) and fold it.
    if node.leaf {
        return false;
    }
    // Prefer reducing deeper first.
    for i in 0..8 {
        if let Some(ch) = node.children[i].as_mut() {
            if !ch.leaf && reduce_tree(ch, leaf_count) {
                return true;
            }
        }
    }
    let mut child_leaves = 0;
    let mut has_branch = false;
    for c in &node.children {
        match c {
            Some(ch) if ch.leaf => child_leaves += 1,
            Some(ch) if !ch.leaf => has_branch = true,
            _ => {}
        }
    }
    if !has_branch && child_leaves > 0 {
        let mut count = 0u32;
        let mut r = 0u64;
        let mut g = 0u64;
        let mut b = 0u64;
        for c in &mut node.children {
            if let Some(ch) = c.take() {
                count += ch.count;
                r += ch.r;
                g += ch.g;
                b += ch.b;
                *leaf_count -= 1;
            }
        }
        node.leaf = true;
        node.count = count;
        node.r = r;
        node.g = g;
        node.b = b;
        *leaf_count += 1;
        return true;
    }
    false
}

fn collect_leaves(node: &OctNode, out: &mut Vec<[u8; 3]>) {
    if node.leaf {
        if node.count > 0 {
            out.push([
                (node.r / node.count as u64) as u8,
                (node.g / node.count as u64) as u8,
                (node.b / node.count as u64) as u8,
            ]);
        }
        return;
    }
    for ch in node.children.iter().flatten() {
        collect_leaves(ch, out);
    }
}

// ---- Median-cut (fallback) --------------------------------------------------

fn median_cut(colors: &[([u8; 3], u32)], max_colors: usize) -> Vec<[u8; 3]> {
    #[derive(Clone)]
    struct Box {
        colors: Vec<([u8; 3], u32)>,
    }
    impl Box {
        fn range_channel(&self) -> usize {
            let mut min = [255u8; 3];
            let mut max = [0u8; 3];
            for (c, _) in &self.colors {
                for i in 0..3 {
                    min[i] = min[i].min(c[i]);
                    max[i] = max[i].max(c[i]);
                }
            }
            let ranges = [
                max[0] as i32 - min[0] as i32,
                max[1] as i32 - min[1] as i32,
                max[2] as i32 - min[2] as i32,
            ];
            ranges
                .iter()
                .enumerate()
                .max_by_key(|&(_, r)| r)
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
        fn average(&self) -> [u8; 3] {
            let mut sum = [0u64; 3];
            let mut n = 0u64;
            for (c, w) in &self.colors {
                let w = *w as u64;
                for i in 0..3 {
                    sum[i] += c[i] as u64 * w;
                }
                n += w;
            }
            if n == 0 {
                return [0, 0, 0];
            }
            [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8]
        }
    }

    let mut boxes = vec![Box {
        colors: colors.to_vec(),
    }];
    while boxes.len() < max_colors {
        let Some(idx) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() >= 2)
            .max_by_key(|(_, b)| {
                let ch = b.range_channel();
                let mut min = 255u8;
                let mut max = 0u8;
                for (c, _) in &b.colors {
                    min = min.min(c[ch]);
                    max = max.max(c[ch]);
                }
                (max as u32).saturating_sub(min as u32)
            })
            .map(|(i, _)| i)
        else {
            break;
        };
        let mut box_ = boxes.swap_remove(idx);
        let ch = box_.range_channel();
        box_.colors.sort_by_key(|(c, _)| c[ch]);
        let mid = box_.colors.len() / 2;
        let right = box_.colors.split_off(mid);
        boxes.push(Box {
            colors: box_.colors,
        });
        boxes.push(Box { colors: right });
    }
    boxes.into_iter().map(|b| b.average()).collect()
}

fn nearest_index(rgb: [u8; 3], palette: &[[u8; 3]], transparent: Option<u8>) -> u8 {
    let limit = transparent.map(|t| t as usize).unwrap_or(palette.len());
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, c) in palette
        .iter()
        .enumerate()
        .take(limit.max(1).min(palette.len()))
    {
        let d = dist2(rgb, *c);
        if d < best_d {
            best_d = d;
            best = i;
            if d == 0 {
                break;
            }
        }
    }
    best as u8
}

fn nearest_index_lossy(
    rgb: [u8; 3],
    palette: &[[u8; 3]],
    transparent: Option<u8>,
    neighbor: Option<u8>,
    lossy: u8,
) -> u8 {
    if lossy > 0 {
        if let Some(index) = neighbor {
            let i = index as usize;
            if i < palette.len()
                && transparent != Some(index)
                && dist2(rgb, palette[i]) <= lossy_dist2(lossy)
            {
                return index;
            }
        }
    }
    nearest_index(rgb, palette, transparent)
}

fn lossy_dist2(lossy: u8) -> u32 {
    let channel = u32::from(lossy) * 255 / 100;
    channel * channel * 3
}

fn dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

fn lzw_encode(indices: &[u8], min_code_size: u8, mode: LzwClearMode) -> Vec<u8> {
    let clear = 1u16 << min_code_size;
    let end = clear + 1;
    let mut code_size = min_code_size as u32 + 1;
    let mut next_code = end + 1;
    let mut dict = HashMap::<u32, u16>::new();
    let mut out_bits = BitPacker::new();
    out_bits.write(clear, code_size);
    if indices.is_empty() {
        out_bits.write(end, code_size);
        return out_bits.finish();
    }
    let mut w = indices[0] as u16;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut frozen_hits = 0usize;
    let mut frozen_misses = 0usize;
    for &k in &indices[1..] {
        let key = ((w as u32) << 8) | k as u32;
        if let Some(&code) = dict.get(&key) {
            hits += 1;
            if next_code >= 4096 {
                frozen_hits += 1;
            }
            w = code;
            continue;
        }
        misses += 1;
        if next_code >= 4096 {
            frozen_misses += 1;
        }
        out_bits.write(w, code_size);
        if next_code < 4096 {
            dict.insert(key, next_code);
            // Bump width when the code we just assigned fills the current
            // alphabet — *before* advancing `next_code` (GIF LZW quirk).
            if next_code == (1 << code_size) && code_size < 12 {
                code_size += 1;
            }
            next_code += 1;
        } else {
            let clear_now = match mode {
                LzwClearMode::WhenFull => true,
                LzwClearMode::Deferred => {
                    frozen_misses >= 256 && frozen_hits.saturating_mul(4) < frozen_misses
                }
                LzwClearMode::Adaptive => frozen_misses >= 128 && frozen_hits < frozen_misses,
            };
            if clear_now {
                out_bits.write(clear, code_size);
                dict.clear();
                code_size = min_code_size as u32 + 1;
                next_code = end + 1;
                hits = 0;
                misses = 0;
                frozen_hits = 0;
                frozen_misses = 0;
            }
        }
        if mode == LzwClearMode::Adaptive
            && next_code < 4096
            && next_code > end + 512
            && misses >= 512
            && misses > hits.saturating_mul(3)
        {
            out_bits.write(clear, code_size);
            dict.clear();
            code_size = min_code_size as u32 + 1;
            next_code = end + 1;
            hits = 0;
            misses = 0;
            frozen_hits = 0;
            frozen_misses = 0;
        }
        w = k as u16;
    }
    out_bits.write(w, code_size);
    out_bits.write(end, code_size);
    out_bits.finish()
}

struct BitPacker {
    buf: Vec<u8>,
    cur: u32,
    bits: u32,
}

impl BitPacker {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur: 0,
            bits: 0,
        }
    }
    fn write(&mut self, code: u16, nbits: u32) {
        self.cur |= (code as u32) << self.bits;
        self.bits += nbits;
        while self.bits >= 8 {
            self.buf.push((self.cur & 0xFF) as u8);
            self.cur >>= 8;
            self.bits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.buf.push((self.cur & 0xFF) as u8);
        }
        self.buf
    }
}

fn write_sub_blocks(out: &mut impl Write, data: &[u8]) -> io::Result<()> {
    let mut i = 0;
    while i < data.len() {
        let n = (data.len() - i).min(255);
        out.write_all(&[n as u8])?;
        out.write_all(&data[i..i + n])?;
        i += n;
    }
    out.write_all(&[0x00])
}

#[cfg(test)]
fn gct_size_from_gif(gif: &[u8]) -> Option<usize> {
    let packed = gif[10];
    if packed & 0x80 == 0 {
        return None;
    }
    Some(1usize << ((packed & 0x07) + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for p in v.chunks_mut(4) {
            p[0] = r;
            p[1] = g;
            p[2] = b;
            p[3] = a;
        }
        v
    }

    #[test]
    fn header_trailer_netscape() {
        let mut enc = GifEncoder::new(4, 4, 0)
            .dither(false)
            .diff_rects(false)
            .palette_mode(PaletteMode::Global)
            .transparency(false);
        enc.add_frame_owned(solid(4, 4, 255, 255, 255, 255), 1, 10);
        enc.add_frame_owned(solid(4, 4, 200, 0, 0, 255), 1, 10);
        let gif = enc.finish();
        assert_eq!(&gif[..6], b"GIF89a");
        assert_eq!(*gif.last().unwrap(), 0x3B);
        assert!(gif.windows(11).any(|w| w == b"NETSCAPE2.0"));
    }

    #[test]
    fn transparency_gce_flag() {
        let mut enc = GifEncoder::new(2, 2, 1)
            .dither(false)
            .diff_rects(false)
            .transparency(true);
        let mut rgba = solid(2, 2, 255, 0, 0, 255);
        rgba[7] = 0;
        enc.add_frame_owned(rgba, 1, 10);
        let gif = enc.finish();
        let gce = gif.windows(2).position(|w| w == [0x21, 0xF9]).unwrap();
        assert_eq!(gif[gce + 3] & 1, 1);
    }

    #[test]
    fn colors_4_global_gct() {
        let mut enc = GifEncoder::new(8, 8, 1)
            .colors(4)
            .dither(false)
            .diff_rects(false)
            .palette_mode(PaletteMode::Global)
            .transparency(false);
        let mut rgba = vec![0u8; 8 * 8 * 4];
        for (i, p) in rgba.chunks_mut(4).enumerate() {
            let c = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]][i % 4];
            p[..3].copy_from_slice(&c);
            p[3] = 255;
        }
        enc.add_frame_owned(rgba, 1, 10);
        let gif = enc.finish();
        assert_eq!(gct_size_from_gif(&gif), Some(4));
    }

    #[test]
    fn dirty_rect_shrinks_second_frame() {
        let w = 32u32;
        let h = 32u32;
        let mut with_diff = GifEncoder::new(w, h, 1)
            .colors(16)
            .dither(false)
            .diff_rects(true)
            .palette_mode(PaletteMode::Local)
            .transparency(true);
        let mut no_diff = GifEncoder::new(w, h, 1)
            .colors(16)
            .dither(false)
            .diff_rects(false)
            .palette_mode(PaletteMode::Local)
            .transparency(true);

        let f0 = solid(w, h, 0, 0, 0, 255);
        let mut f1 = solid(w, h, 0, 0, 0, 255);
        for y in 10..14 {
            for x in 10..14 {
                let i = ((y * w + x) * 4) as usize;
                f1[i] = 255;
            }
        }
        with_diff.add_frame_owned(f0.clone(), 1, 10);
        with_diff.add_frame_owned(f1.clone(), 1, 10);
        no_diff.add_frame_owned(f0, 1, 10);
        no_diff.add_frame_owned(f1, 1, 10);

        let a = with_diff.finish();
        let b = no_diff.finish();
        assert!(
            a.len() < b.len(),
            "diff-rect GIF should be smaller: {} vs {}",
            a.len(),
            b.len()
        );

        // Parse image descriptor sizes correctly (skip LCT / LZW).
        let sizes = image_sizes(&a);
        assert!(sizes.len() >= 2, "expected ≥2 images, got {sizes:?}");
        assert_eq!(sizes[0], (32, 32));
        assert!(
            sizes[1].0 <= 8 && sizes[1].1 <= 8,
            "dirty rect should be small, got {:?}",
            sizes[1]
        );
    }

    /// Walk GIF89a and collect `(width, height)` from each Image Descriptor.
    fn image_sizes(gif: &[u8]) -> Vec<(u16, u16)> {
        let mut out = Vec::new();
        let mut i = 13; // after header + LSD (no GCT when local-only stream)
                        // Skip GCT if present.
        let packed = gif[10];
        if packed & 0x80 != 0 {
            let n = 1usize << ((packed & 7) + 1);
            i = 13 + n * 3;
        }
        while i < gif.len() {
            match gif[i] {
                0x3B => break,
                0x21 => {
                    // Extension.
                    i += 1;
                    if i >= gif.len() {
                        break;
                    }
                    let label = gif[i];
                    i += 1;
                    if label == 0xFF || label == 0xF9 || label == 0xFE || label == 0x01 {
                        while i < gif.len() {
                            let sz = gif[i] as usize;
                            i += 1;
                            if sz == 0 {
                                break;
                            }
                            i += sz;
                        }
                    }
                }
                0x2C => {
                    if i + 10 > gif.len() {
                        break;
                    }
                    let iw = u16::from_le_bytes([gif[i + 5], gif[i + 6]]);
                    let ih = u16::from_le_bytes([gif[i + 7], gif[i + 8]]);
                    out.push((iw, ih));
                    let ipacked = gif[i + 9];
                    i += 10;
                    if ipacked & 0x80 != 0 {
                        let n = 1usize << ((ipacked & 7) + 1);
                        i += n * 3;
                    }
                    // LZW: min code size + sub-blocks.
                    i += 1;
                    while i < gif.len() {
                        let sz = gif[i] as usize;
                        i += 1;
                        if sz == 0 {
                            break;
                        }
                        i += sz;
                    }
                }
                _ => i += 1,
            }
        }
        out
    }

    #[test]
    fn streaming_encode_iter() {
        let frames = (0..3).map(|f| {
            let mut rgba = solid(8, 8, 10, 20, 30, 255);
            rgba[0] = (f * 40) as u8;
            (rgba, 1u16, 10u16)
        });
        let gif = GifEncoder::new(8, 8, 0)
            .colors(32)
            .dither(true)
            .diff_rects(true)
            .palette_mode(PaletteMode::Local)
            .encode_iter(frames);
        assert_eq!(&gif[..6], b"GIF89a");
    }

    #[test]
    fn lossy_zero_is_identical() {
        let frames = vec![
            (solid(8, 8, 10, 20, 30, 255), 1, 10),
            (solid(8, 8, 11, 21, 31, 255), 1, 10),
        ];
        let baseline = encode_gif(8, 8, 0, &GifOptions::default(), frames.clone());
        let explicit = encode_gif(8, 8, 0, &GifOptions::default().lossy(0), frames);
        assert_eq!(baseline, explicit);
    }

    #[test]
    fn gif_writer_matches_local_encode() {
        let frames = vec![
            (solid(4, 4, 20, 30, 40, 255), 1, 10),
            (solid(4, 4, 50, 60, 70, 255), 2, 10),
        ];
        let expected = encode_gif(4, 4, 0, &GifOptions::default(), frames.clone());
        let mut writer = GifWriter::new(Vec::new(), 4, 4, 0, GifOptions::default()).unwrap();
        for (rgba, n, d) in frames {
            writer.write_frame(&rgba, n, d).unwrap();
        }
        assert_eq!(writer.finish().unwrap(), expected);
    }

    #[test]
    fn comment_extension_is_emitted() {
        let opts = GifOptions::default().comment(b"hello");
        let gif = encode_gif(1, 1, 1, &opts, vec![(solid(1, 1, 0, 0, 0, 255), 1, 10)]);
        assert!(gif.windows(2).any(|w| w == [0x21, 0xFE]));
        assert!(gif.windows(5).any(|w| w == b"hello"));
    }

    #[test]
    fn interlace_descriptor_bit_is_set() {
        let opts = GifOptions::default().interlace(true);
        let gif = encode_gif(2, 2, 1, &opts, vec![(solid(2, 2, 1, 2, 3, 255), 1, 10)]);
        let image = gif.iter().position(|&b| b == 0x2C).unwrap();
        assert_ne!(gif[image + 9] & 0x40, 0);
    }

    #[test]
    fn alternate_lzw_clear_modes_form_gifs() {
        for mode in [LzwClearMode::Deferred, LzwClearMode::Adaptive] {
            let opts = GifOptions::default().lzw_clear(mode);
            let gif = encode_gif(
                32,
                32,
                1,
                &opts,
                vec![(solid(32, 32, 12, 34, 56, 255), 1, 10)],
            );
            assert_eq!(&gif[..6], b"GIF89a");
            assert_eq!(gif.last(), Some(&0x3B));
        }
    }

    #[test]
    fn auto_disposal_handles_transparent_hole() {
        let opaque = solid(3, 3, 255, 0, 0, 255);
        let mut hole = opaque.clone();
        hole[(4 * 4) + 3] = 0;
        let gif = encode_gif(
            3,
            3,
            1,
            &GifOptions::default().dither(false),
            vec![(opaque, 1, 10), (hole, 1, 10)],
        );
        let gce = gif.windows(2).position(|w| w == [0x21, 0xF9]).unwrap();
        let disposal = (gif[gce + 3] >> 2) & 7;
        assert!(disposal == 2 || disposal == 3, "got disposal {disposal}");
    }

    #[test]
    fn octree_reduces_colors() {
        let mut samples = Vec::new();
        for r in (0..256).step_by(2) {
            for g in (0..256).step_by(4) {
                samples.push([r as u8, g as u8, 128]);
            }
        }
        let pal = octree_quantize(&samples, 16);
        assert!(pal.len() <= 16, "got {}", pal.len());
        assert!(!pal.is_empty());
    }
}
