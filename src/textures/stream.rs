//! Textures whose pixels change while the texture stays the same: video, a
//! camera feed, anything drawn on the CPU every frame.
//!
//! A material holds its maps as `Arc<Texture>`, and the renderer caches what it
//! uploads, and the bind groups around it, by the texture's identity. Changing
//! the pixels through [`Texture::replace_rgba_bytes`] means a new `Texture` in
//! the material every frame and a new GPU texture behind it. A
//! [`TextureStream`] avoids both: the texture carries the stream, a producer
//! writes frames into it from any thread, and the renderer copies the newest
//! frame into the GPU texture it already has.
//!
//! [`VideoTexture`] is the usual way in:
//!
//! ```
//! use threers::textures::VideoTexture;
//! use threers::materials::{BasicMaterial, Material};
//! use threers::math::Color;
//!
//! let video = VideoTexture::new(320, 180);
//! let mut screen = BasicMaterial::new(Color::WHITE);
//! screen.map = Some(video.texture());
//! let material = Material::Basic(screen);
//! // Each new frame, from this thread or another:
//! video.write_rgba(&vec![255; 320 * 180 * 4]);
//! # let _ = material;
//! ```

use super::{Texture, TextureFilter, TextureFormat};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The YCbCr → RGB matrix a [`YuvFrame`] was coded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YuvMatrix {
    /// ITU-R BT.601 (SD).
    Bt601,
    /// ITU-R BT.709 (HD), the usual one.
    #[default]
    Bt709,
    /// ITU-R BT.2020 (UHD), non-constant luminance.
    Bt2020,
}

/// One plane's samples: a byte each (8-bit video), or a `u16` each (any
/// depth, as the decoders give them).
#[derive(Debug, Clone)]
pub enum Samples {
    /// One byte a sample: half the upload of `Words`.
    Bytes(Vec<u8>),
    /// Two bytes a sample.
    Words(Vec<u16>),
}

impl Default for Samples {
    fn default() -> Self {
        Samples::Words(Vec::new())
    }
}

impl From<Vec<u8>> for Samples {
    fn from(v: Vec<u8>) -> Self {
        Samples::Bytes(v)
    }
}

impl From<Vec<u16>> for Samples {
    fn from(v: Vec<u16>) -> Self {
        Samples::Words(v)
    }
}

impl Samples {
    /// How many samples.
    pub fn len(&self) -> usize {
        match self {
            Samples::Bytes(v) => v.len(),
            Samples::Words(v) => v.len(),
        }
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample `i`.
    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        match self {
            Samples::Bytes(v) => v[i] as u32,
            Samples::Words(v) => v[i] as u32,
        }
    }

    /// 1 or 2.
    pub fn bytes_per_sample(&self) -> u32 {
        match self {
            Samples::Bytes(_) => 1,
            Samples::Words(_) => 2,
        }
    }

    /// The samples as bytes, native-endian: what is uploaded.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Samples::Bytes(v) => v,
            Samples::Words(v) => bytemuck::cast_slice(v),
        }
    }
}

/// A picture as a video decoder gives it: a luma plane and two chroma planes,
/// top row first.
///
/// Writing these into a [`TextureStream`] rather than RGBA lets the GPU
/// renderers do the colour conversion, so no CPU pass touches every pixel:
/// at 4K and 8K that pass was most of the cost of showing a frame. 8-bit
/// video as [`Samples::Bytes`] uploads 1.5 bytes a pixel at 4:2:0, against
/// 4 for RGBA.
#[derive(Debug, Clone, Default)]
pub struct YuvFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bits a sample, 8 to 16.
    pub bit_depth: u8,
    /// Chroma subsampling, across and down: `(2, 2)` for 4:2:0, `(2, 1)` for
    /// 4:2:2, `(1, 1)` for 4:4:4, `(1, 2)` for 4:4:0.
    pub subsampling: (u8, u8),
    /// Luma, `width * height`.
    pub y: Samples,
    /// Cb, [`chroma_size`](Self::chroma_size); empty for monochrome. The
    /// same kind of [`Samples`] as `y`.
    pub u: Samples,
    /// Cr, laid out as `u`; empty when `interleaved`.
    pub v: Samples,
    /// `u` holds Cb and Cr in pairs (NV12 and P010 order, as hardware
    /// decoders give them) and `v` is empty. The GPU reads the pairs as they
    /// are, so a hardware decoder's picture needs no pass on the CPU.
    pub interleaved: bool,
    /// The conversion matrix.
    pub matrix: YuvMatrix,
    /// Full range (0–255 at 8 bits) rather than video range (16–235).
    pub full_range: bool,
}

/// The arithmetic of a [`YuvFrame`]'s conversion, in output units of 0 to 1:
/// `rgb = y_scale (Y - y_off) + M (C - c_mid)`. The GPU passes take it as
/// it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YuvCoefficients {
    /// Black's luma code.
    pub y_off: f32,
    /// Luma code to 0..1.
    pub y_scale: f32,
    /// The chroma code for no colour.
    pub c_mid: f32,
    /// Cr into red.
    pub rv: f32,
    /// Cb out of green.
    pub gu: f32,
    /// Cr out of green.
    pub gv: f32,
    /// Cb into blue.
    pub bu: f32,
}

impl YuvFrame {
    /// Whether it is luma only.
    pub fn is_monochrome(&self) -> bool {
        self.u.is_empty()
    }

    /// The size of each chroma plane; `(0, 0)` for monochrome.
    pub fn chroma_size(&self) -> (u32, u32) {
        if self.is_monochrome() {
            return (0, 0);
        }
        let (sw, sh) = self.subsampling;
        (
            self.width.div_ceil(sw.max(1) as u32),
            self.height.div_ceil(sh.max(1) as u32),
        )
    }

    /// Whether the planes hold as many samples as the sizes say, and the
    /// subsampling and depth are ones the converters know.
    pub fn is_valid(&self) -> bool {
        let (cw, ch) = self.chroma_size();
        let chroma = cw as usize * ch as usize;
        let bytes = self.y.bytes_per_sample();
        if self.interleaved && !(self.v.is_empty() && self.u.len() == 2 * chroma) {
            return false;
        }
        matches!(self.subsampling, (1 | 2, 1 | 2))
            && (8..=16).contains(&self.bit_depth)
            && (bytes == 2 || self.bit_depth == 8)
            && (self.u.is_empty() || self.u.bytes_per_sample() == bytes)
            && (self.v.is_empty() || self.v.bytes_per_sample() == bytes)
            && self.y.len() == self.width as usize * self.height as usize
            && (self.interleaved || (self.u.len() == self.v.len() && (self.u.is_empty() || self.u.len() == chroma)))
    }

    /// The conversion's constants for this frame's depth, range and matrix.
    pub fn coefficients(&self) -> YuvCoefficients {
        let depth = self.bit_depth.clamp(8, 16) as u32;
        let max = ((1u64 << depth) - 1) as f64;
        let s = (1u64 << (depth - 8)) as f64;
        let (y_off, y_scale, c_scale) = if self.full_range {
            (0.0, 1.0 / max, 1.0 / max)
        } else {
            (16.0 * s, 1.0 / (219.0 * s), 1.0 / (224.0 * s))
        };
        let (kr, kb) = match self.matrix {
            YuvMatrix::Bt601 => (0.299, 0.114),
            YuvMatrix::Bt709 => (0.2126, 0.0722),
            YuvMatrix::Bt2020 => (0.2627, 0.0593),
        };
        let kg = 1.0 - kr - kb;
        YuvCoefficients {
            y_off: y_off as f32,
            y_scale: y_scale as f32,
            c_mid: (1u64 << (depth - 1)) as f32,
            rv: (2.0 * (1.0 - kr) * c_scale) as f32,
            gu: (2.0 * kb * (1.0 - kb) / kg * c_scale) as f32,
            gv: (2.0 * kr * (1.0 - kr) / kg * c_scale) as f32,
            bu: (2.0 * (1.0 - kb) * c_scale) as f32,
        }
    }

    /// Convert to 8-bit RGBA in `out`, top row first, on this thread: the
    /// same sums the GPU passes do. For what reads a stream on the CPU (the
    /// path tracer, exporters, tests); the renderers convert on the GPU.
    ///
    /// Chroma is upsampled bilinearly from where H.264, HEVC and AV1 site it
    /// by default: level with the even luma columns, and halfway between luma
    /// rows when halved vertically.
    pub fn to_rgba_into(&self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        out.clear();
        out.resize(w * h * 4, 255);
        if !self.is_valid() {
            return;
        }
        let k = self.coefficients();
        let (cw, ch) = self.chroma_size();
        let (cw, ch) = (cw as usize, ch as usize);
        let (sw, sh) = (self.subsampling.0 == 2, self.subsampling.1 == 2);
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        for y in 0..h {
            let rows = chroma_rows(y, ch, sh);
            for x in 0..w {
                let (cb, cr) = if self.is_monochrome() {
                    (0.0, 0.0)
                } else {
                    let cols = chroma_cols(x, cw, sw);
                    // Sample `i` of Cb (0) or Cr (1), planar or in pairs.
                    let at = |which: usize, i: usize| {
                        if self.interleaved {
                            self.u.get(2 * i + which)
                        } else if which == 0 {
                            self.u.get(i)
                        } else {
                            self.v.get(i)
                        }
                    };
                    let tap = |which: usize| {
                        let mut sum = 0u32;
                        for (r, wr) in [(rows.0, rows.2), (rows.1, rows.3)] {
                            for (c, wc) in [(cols.0, cols.2), (cols.1, cols.3)] {
                                sum += wr * wc * at(which, r * cw + c);
                            }
                        }
                        sum as f32 / 8.0 - k.c_mid
                    };
                    (tap(0), tap(1))
                };
                let l = k.y_scale * (self.y.get(y * w + x) as f32 - k.y_off);
                let px = &mut out[(y * w + x) * 4..][..3];
                px[0] = to8(l + k.rv * cr);
                px[1] = to8(l - k.gu * cb - k.gv * cr);
                px[2] = to8(l + k.bu * cb);
            }
        }
    }
}

/// The two chroma rows luma row `y` reads and their weights, out of 4.
fn chroma_rows(y: usize, ch: usize, halved: bool) -> (usize, usize, u32, u32) {
    if !halved {
        let r = y.min(ch.saturating_sub(1));
        return (r, r, 4, 0);
    }
    let c = (y / 2).min(ch.saturating_sub(1));
    let other = if y.is_multiple_of(2) { c.saturating_sub(1) } else { (c + 1).min(ch.saturating_sub(1)) };
    (c, other, 3, 1)
}

/// The two chroma columns luma column `x` reads and their weights, out of 2.
fn chroma_cols(x: usize, cw: usize, halved: bool) -> (usize, usize, u32, u32) {
    if !halved {
        let c = x.min(cw.saturating_sub(1));
        return (c, c, 2, 0);
    }
    let j = x / 2;
    if x.is_multiple_of(2) {
        (j, j, 2, 0)
    } else {
        (j, (j + 1).min(cw.saturating_sub(1)), 1, 1)
    }
}

/// What a [`TextureStream`] holds now, for a renderer to upload.
#[derive(Debug, Clone, Copy)]
pub enum StreamFrame<'a> {
    /// RGBA8, `width * height * 4` bytes, top row first.
    Rgba(&'a [u8]),
    /// Planes for the GPU to convert.
    Yuv(&'a YuvFrame),
    /// A picture the browser holds, decoded by it (in hardware where it can),
    /// for the renderer to copy on the GPU.
    #[cfg(target_arch = "wasm32")]
    Browser(&'a BrowserImage),
}

/// A picture the browser decoded and holds: a playing `<video>`'s current
/// frame, or a WebCodecs `VideoFrame`. The wgpu renderer copies it into the
/// texture GPU to GPU (`copyExternalImageToTexture`), so it never comes back
/// to the CPU: what makes 4K and 8K video usable in a browser.
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone)]
pub enum BrowserImage {
    /// A video element; whatever frame it shows when the renderer copies.
    Video(web_sys::HtmlVideoElement),
    /// A decoded frame. The stream owns it and closes it when it is replaced,
    /// returning its memory to the decoder.
    Frame(web_sys::VideoFrame),
}

#[cfg(target_arch = "wasm32")]
impl BrowserImage {
    /// Its size in pixels.
    pub fn size(&self) -> (u32, u32) {
        match self {
            BrowserImage::Video(v) => (v.video_width(), v.video_height()),
            BrowserImage::Frame(f) => (f.display_width(), f.display_height()),
        }
    }

    fn release(&self) {
        if let BrowserImage::Frame(f) = self {
            f.close();
        }
    }
}

/// A browser object held in a stream. JavaScript objects belong to the
/// thread that made them, and a stream is only written and read on the
/// browser's main thread (the one the renderer runs on), so it is never
/// touched from another.
#[cfg(target_arch = "wasm32")]
struct OnMainThread(BrowserImage);
#[cfg(target_arch = "wasm32")]
unsafe impl Send for OnMainThread {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for OnMainThread {}
#[cfg(target_arch = "wasm32")]
impl Drop for OnMainThread {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// The frame itself, and an RGBA copy of a YUV one made for a CPU reader.
struct Held {
    frame: Kind,
    /// `read` on a YUV frame: converted once per frame, kept until the next.
    rgba: Option<(u64, Vec<u8>)>,
}

enum Kind {
    Rgba(Vec<u8>),
    Yuv(YuvFrame),
    #[cfg(target_arch = "wasm32")]
    Browser(OnMainThread),
}


/// The latest frame for a streamed [`Texture`], and how many have been written.
///
/// Frames are RGBA8 in the texture's size, top row first, or [`YuvFrame`]
/// planes the renderer converts. Writing replaces the frame; a renderer that
/// misses some shows the newest.
pub struct TextureStream {
    width: u32,
    height: u32,
    held: Mutex<Held>,
    seq: AtomicU64,
}

impl std::fmt::Debug for TextureStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureStream")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frames", &self.frames())
            .finish()
    }
}

impl TextureStream {
    /// A stream of `width` by `height` frames, black and opaque until the
    /// first is written. Nothing the size of a frame is allocated until then.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            held: Mutex::new(Held {
                frame: Kind::Rgba(Vec::new()),
                rgba: None,
            }),
            seq: AtomicU64::new(0),
        }
    }

    /// Frame size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// How many frames have been written. A renderer compares this with the
    /// count it last uploaded.
    pub fn frames(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replace the frame with `rgba`, `width * height * 4` bytes, top row
    /// first. Returns false, changing nothing, when the length is wrong.
    pub fn write_rgba(&self, rgba: &[u8]) -> bool {
        self.write_with(|frame| {
            if frame.len() != rgba.len() {
                return false;
            }
            frame.copy_from_slice(rgba);
            true
        })
    }

    /// Fill the frame in place, for a producer that converts straight into it
    /// instead of building a buffer first. `fill` gets `width * height * 4`
    /// bytes and returns whether it wrote a frame; only then does the count go
    /// up.
    pub fn write_with(&self, fill: impl FnOnce(&mut [u8]) -> bool) -> bool {
        let mut held = self.lock();
        if !matches!(held.frame, Kind::Rgba(_)) {
            // Back to RGBA: reuse the copy `read` made, if there is one.
            let buffer = held.rgba.take().map(|(_, b)| b);
            let size = self.width as usize * self.height as usize * 4;
            held.frame = Kind::Rgba(buffer.filter(|b| b.len() == size).unwrap_or_else(|| vec![0; size]));
        }
        let Kind::Rgba(frame) = &mut held.frame else { unreachable!() };
        let size = self.width as usize * self.height as usize * 4;
        if frame.len() != size {
            *frame = black(self.width, self.height);
        }
        let wrote = fill(frame);
        if wrote {
            self.seq.fetch_add(1, Ordering::Release);
        }
        wrote
    }

    /// Replace the frame with YUV planes, for the renderer to convert. The
    /// frame is moved in, not copied. Returns false, changing nothing, when
    /// its size is not the stream's or its planes do not match it.
    pub fn write_yuv(&self, frame: YuvFrame) -> bool {
        self.swap_yuv(frame).is_ok()
    }

    /// [`write_yuv`](Self::write_yuv), handing back the YUV frame it
    /// replaced (`None` when there was none), for its buffers to be filled
    /// again: see `codec::bytes::FramePool`. The frame comes back as `Err`
    /// when it was refused.
    pub fn swap_yuv(&self, frame: YuvFrame) -> Result<Option<YuvFrame>, YuvFrame> {
        if (frame.width, frame.height) != (self.width, self.height) || !frame.is_valid() {
            return Err(frame);
        }
        let mut held = self.lock();
        let replaced = match std::mem::replace(&mut held.frame, Kind::Yuv(frame)) {
            Kind::Rgba(old) => {
                // Kept for `read` to convert into.
                if held.rgba.is_none() && !old.is_empty() {
                    held.rgba = Some((u64::MAX, old));
                }
                None
            }
            Kind::Yuv(old) => Some(old),
            #[cfg(target_arch = "wasm32")]
            Kind::Browser(_) => None,
        };
        self.seq.fetch_add(1, Ordering::Release);
        Ok(replaced)
    }

    /// Show a picture the browser decoded: a `<video>` (its current frame,
    /// whenever the renderer copies) or a WebCodecs `VideoFrame` (owned from
    /// here on, and closed when replaced). The renderer copies it GPU to GPU.
    /// False, changing nothing, when its size is not the stream's.
    #[cfg(target_arch = "wasm32")]
    pub fn write_browser(&self, image: BrowserImage) -> bool {
        if image.size() != (self.width, self.height) {
            image.release();
            return false;
        }
        self.lock().frame = Kind::Browser(OnMainThread(image));
        self.seq.fetch_add(1, Ordering::Release);
        true
    }

    /// Run `read` on the current frame as RGBA, and the count it belongs to.
    /// A YUV frame is converted on this thread first, once per frame. Holds
    /// the frame for the call, so a writer waits rather than tearing it.
    pub fn read<R>(&self, read: impl FnOnce(&[u8], u64) -> R) -> R {
        let mut held = self.lock();
        let seq = self.seq.load(Ordering::Acquire);
        let Held { frame, rgba } = &mut *held;
        match frame {
            Kind::Rgba(bytes) if bytes.is_empty() => {
                // Nothing written yet: black, made for the reader.
                if rgba.is_none() {
                    *rgba = Some((u64::MAX, black(self.width, self.height)));
                }
                read(&rgba.as_ref().expect("just made").1, seq)
            }
            Kind::Rgba(bytes) => read(bytes, seq),
            // The browser's picture cannot be read back here without waiting
            // on the GPU: readers on the CPU see black.
            #[cfg(target_arch = "wasm32")]
            Kind::Browser(_) => {
                let (w, h) = (self.width, self.height);
                let buffer = rgba.get_or_insert_with(|| (u64::MAX, black(w, h)));
                if buffer.0 != seq {
                    *buffer = (seq, black(w, h));
                }
                read(&buffer.1, seq)
            }
            Kind::Yuv(yuv) => {
                let (at, buffer) = rgba.get_or_insert_with(|| (u64::MAX, Vec::new()));
                if *at != seq {
                    yuv.to_rgba_into(buffer);
                    *at = seq;
                }
                read(buffer, seq)
            }
        }
    }

    /// Run `read` on the current frame as written, RGBA or YUV, and the count
    /// it belongs to: what a renderer uploads. Before the first frame is
    /// written the RGBA is empty: nothing to upload, the texture stays black.
    pub fn read_frame<R>(&self, read: impl FnOnce(StreamFrame<'_>, u64) -> R) -> R {
        let held = self.lock();
        let seq = self.seq.load(Ordering::Acquire);
        match &held.frame {
            Kind::Rgba(bytes) => read(StreamFrame::Rgba(bytes), seq),
            Kind::Yuv(yuv) => read(StreamFrame::Yuv(yuv), seq),
            #[cfg(target_arch = "wasm32")]
            Kind::Browser(image) => read(StreamFrame::Browser(&image.0), seq),
        }
    }
}

/// An opaque black RGBA frame.
fn black(width: u32, height: u32) -> Vec<u8> {
    let mut black = vec![0u8; width as usize * height as usize * 4];
    black.chunks_exact_mut(4).for_each(|p| p[3] = 255);
    black
}

/// A texture fed with frames: put [`texture`](Self::texture) in a material,
/// then write frames as they come. Cloning shares the stream.
///
/// Colour frames, so the texture is sRGB, with linear filtering and a mip
/// chain the GPU renderers rebuild after each frame, so a 4K picture on a
/// small screen does not shimmer. Turn the chain off with
/// [`without_mipmaps`](Self::without_mipmaps).
#[derive(Debug, Clone)]
pub struct VideoTexture {
    texture: Arc<Texture>,
    stream: Arc<TextureStream>,
}

impl VideoTexture {
    /// A `width` by `height` texture showing black until the first frame.
    pub fn new(width: u32, height: u32) -> Self {
        let stream = Arc::new(TextureStream::new(width, height));
        // No pixels of its own: everything that reads them goes through the
        // stream ([`Texture::pixels`]), so an 8K frame is not held twice.
        let mut texture = Texture::new(width, height, TextureFormat::Rgba8UnormSrgb, Vec::new());
        texture.mipmap_filter = Some(TextureFilter::Linear);
        texture.min_filter = TextureFilter::Linear;
        texture.mag_filter = TextureFilter::Linear;
        // Frames arrive top row first, as decoders and images give them.
        texture.flip_y = true;
        texture.stream = Some(stream.clone());
        Self {
            texture: Arc::new(texture),
            stream,
        }
    }

    /// The same stream with no mip chain: sampled at full size however small
    /// it is drawn, as three.js's `VideoTexture` is. Saves rebuilding the
    /// chain each frame where the picture is never minified.
    pub fn without_mipmaps(self) -> Self {
        let mut texture = (*self.texture).clone();
        texture.mipmap_filter = None;
        // Another texture to the renderers' caches, made with no chain.
        texture.id = super::texture::next_texture_id();
        Self {
            texture: Arc::new(texture),
            stream: self.stream,
        }
    }

    /// The texture, for a material's `map` (or `emissive_map`, for a screen
    /// that glows).
    pub fn texture(&self) -> Arc<Texture> {
        self.texture.clone()
    }

    /// The stream the frames go into.
    pub fn stream(&self) -> &Arc<TextureStream> {
        &self.stream
    }

    /// Frame size in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.stream.size()
    }

    /// Show `rgba`, `width * height * 4` bytes, top row first. False if the
    /// length is wrong.
    pub fn write_rgba(&self, rgba: &[u8]) -> bool {
        self.stream.write_rgba(rgba)
    }

    /// Show a decoder's planes, converted on the GPU. False if the size or
    /// the planes are wrong.
    pub fn write_yuv(&self, frame: YuvFrame) -> bool {
        self.stream.write_yuv(frame)
    }
}
