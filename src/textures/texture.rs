use crate::math::Vector2;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// Standard 8-bit sRGB color (e.g. baseColor / albedo).
    Rgba8UnormSrgb,
    /// Linear 8-bit (e.g. roughness, metalness, normal maps).
    Rgba8Unorm,
    /// Single-channel 8-bit.
    R8Unorm,
    /// Linear 16-bit float RGBA (8 bytes/px), stored as raw half-float bytes.
    ///
    /// This is the HDR path: unlike the 8-bit formats it can carry values above
    /// 1.0, which is what an environment map needs if it is to hold a sun
    /// alongside a sky (a real ratio of ~10^4:1). Use with a tone-mapped
    /// renderer — see [`crate::renderer::ToneMapping`].
    Rgba16Float,
    /// One 16-bit float channel (2 bytes/px): a height or density field at
    /// half the memory of `R32Float`, filterable everywhere.
    R16Float,
    /// One 32-bit float channel (4 bytes/px): data that must keep f32
    /// precision — heights, distances, ids up to 2^24. Filtering it needs
    /// `wgpu::Features::FLOAT32_FILTERABLE`; without it, sample it nearest.
    R32Float,
    /// Four 32-bit float channels (16 bytes/px): positions, or HDR colour
    /// that must not round to half. Filterable as `R32Float`.
    Rgba32Float,
    /// BC1 ("DXT1"), sRGB. 4x4 blocks in 8 bytes — half a byte a pixel, an
    /// eighth of `Rgba8UnormSrgb`.
    ///
    /// Two RGB565 endpoints and a 2-bit index per pixel, so a block can express
    /// four colours along one line in RGB. That is plenty for photographic
    /// ground and poor for smooth wide gradients, where the four steps become
    /// visible banding. Alpha is 1 throughout: BC1's punch-through mode is not
    /// used here.
    ///
    /// The point of it is size. Blue Marble at its native 500 m is 86400x43200,
    /// which is 13.9 GB as RGBA and will not fit anything; as BC1 it is 1.74 GB,
    /// or 2.32 with a full mip chain.
    Bc1RgbaUnormSrgb,
    /// BC7, sRGB. 4x4 blocks in 16 bytes — one byte a pixel, a quarter of
    /// `Rgba8UnormSrgb`.
    ///
    /// Eight block modes with variable partitioning and endpoint precision, so
    /// it holds gradients that BC1 bands. No encoder here — supply the blocks —
    /// but the format, upload path and mip handling all work, so an externally
    /// encoded map can be used as-is.
    Bc7RgbaUnormSrgb,
}

impl TextureFormat {
    /// Block-compressed formats store a `w x h` block of pixels as one fixed
    /// unit and cannot be addressed per pixel at all.
    pub fn is_block_compressed(self) -> bool {
        matches!(self, Self::Bc1RgbaUnormSrgb | Self::Bc7RgbaUnormSrgb)
    }

    /// Pixels per block, `(1, 1)` for uncompressed formats.
    pub fn block_dim(self) -> (u32, u32) {
        match self {
            Self::Bc1RgbaUnormSrgb | Self::Bc7RgbaUnormSrgb => (4, 4),
            _ => (1, 1),
        }
    }

    /// Bytes per block — per pixel, for the uncompressed formats.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::Rgba8UnormSrgb | Self::Rgba8Unorm => 4,
            Self::R8Unorm => 1,
            Self::Rgba16Float => 8,
            Self::R16Float => 2,
            Self::R32Float => 4,
            Self::Rgba32Float => 16,
            Self::Bc1RgbaUnormSrgb => 8,
            Self::Bc7RgbaUnormSrgb => 16,
        }
    }

    /// Bytes one `width x height` image occupies in this format.
    ///
    /// Block formats round UP to whole blocks, which is why a 4x4-block mip
    /// chain does not simply halve in size at the bottom: a 2x2 level still
    /// costs a whole block, and a 1x1 level costs one too.
    pub fn data_len(self, width: u32, height: u32) -> usize {
        let (bw, bh) = self.block_dim();
        let blocks = (width.max(1).div_ceil(bw) as usize) * (height.max(1).div_ceil(bh) as usize);
        blocks * self.bytes_per_block()
    }

    /// Bytes per row of blocks, for `ImageDataLayout::bytes_per_row`.
    pub fn bytes_per_row(self, width: u32) -> u32 {
        let (bw, _) = self.block_dim();
        width.max(1).div_ceil(bw) * self.bytes_per_block() as u32
    }

    /// Rows of blocks, for `ImageDataLayout::rows_per_image`.
    pub fn rows_per_image(self, height: u32) -> u32 {
        let (_, bh) = self.block_dim();
        height.max(1).div_ceil(bh)
    }

    /// Whether texels are floats (any width), which hold values outside
    /// `[0, 1]` and are not colour-encoded.
    pub fn is_float(self) -> bool {
        matches!(self, Self::Rgba16Float | Self::R16Float | Self::R32Float | Self::Rgba32Float)
    }

    /// Channels a texel holds.
    pub fn channels(self) -> usize {
        match self {
            Self::R8Unorm | Self::R16Float | Self::R32Float => 1,
            _ => 4,
        }
    }

    /// Whether linear filtering needs `wgpu::Features::FLOAT32_FILTERABLE`
    /// (32-bit floats; WebGL2 needs `OES_texture_float_linear` for it).
    pub fn needs_float32_filterable(self) -> bool {
        matches!(self, Self::R32Float | Self::Rgba32Float)
    }

    /// Texels of uncompressed `data` in this format as f32s, every channel
    /// (`R8Unorm` and the 8-bit formats as `byte / 255`, sRGB not decoded).
    /// `None` for the block formats.
    pub fn texels_f32(self, data: &[u8]) -> Option<Vec<f32>> {
        Some(match self {
            Self::Rgba8UnormSrgb | Self::Rgba8Unorm | Self::R8Unorm => data.iter().map(|b| *b as f32 / 255.0).collect(),
            Self::Rgba16Float | Self::R16Float => crate::math::float::unpack_f16(data),
            Self::R32Float | Self::Rgba32Float => crate::math::float::unpack_f32(data),
            Self::Bc1RgbaUnormSrgb | Self::Bc7RgbaUnormSrgb => return None,
        })
    }
}

/// Pack linear f32 RGBA into the raw half-float bytes a
/// [`TextureFormat::Rgba16Float`] texture expects.
///
/// Delegates to the same f32→f16 converter the PMREM atlas upload uses, so
/// there is exactly one rounding implementation in the crate.
pub fn pack_rgba16f(rgba: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba.len() * 2);
    for v in rgba {
        out.extend_from_slice(&crate::renderer::gpu_texture::f32_to_f16_bits(*v).to_le_bytes());
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureWrap {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

/// 2D texture source. Mirrors three.js's `Texture` for the parts we need.
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    /// three.js's mipmapped `minFilter`s, the part [`min_filter`](Self::min_filter)
    /// leaves out: how minification moves between mip levels — `Some(Linear)`
    /// (`*MipmapLinearFilter`, three.js's default), `Some(Nearest)`
    /// (`*MipmapNearestFilter`) — or `None`, no mipmaps at all (`LinearFilter`
    /// / `NearestFilter`), sampling level 0 only.
    pub mipmap_filter: Option<TextureFilter>,
    pub wrap_s: TextureWrap,
    pub wrap_t: TextureWrap,
    /// Pixel data. Length must equal `width * height * bytes_per_pixel`.
    pub data: Arc<Vec<u8>>,
    pub flip_y: bool,
    /// UV transform — translation, rotation, scale all collapsed into a 2D
    /// affine. Mirrors three.js's `Texture.matrix`.
    pub offset: Vector2,
    pub repeat: Vector2,
    pub rotation: f32,
    /// three.js `Texture.center`: the point `rotation` turns about, in UV
    /// space (default `(0, 0)`, the lower-left corner).
    pub center: Vector2,
    /// three.js `Texture.channel` (r151+): which UV set the map reads — 0 is
    /// the geometry's `uv`, 1 its `uv1`. Channels the geometry does not carry
    /// read zero, as an unbound attribute does in WebGL.
    pub channel: u32,
    /// three.js `Texture.matrixAutoUpdate` (default `true`): the UV transform
    /// is rebuilt from `offset`/`repeat`/`rotation`/`center` each frame.
    /// `false` uses [`matrix`](Self::matrix) as given.
    pub matrix_auto_update: bool,
    /// three.js `Texture.matrix` — column-major `Matrix3` elements, read only
    /// when `matrix_auto_update` is off.
    pub matrix: [f32; 9],
    /// If set, this texture is backed by a RenderTarget's color attachment
    /// instead of the CPU `data` bytes. The renderer skips upload and uses
    /// the RenderTarget's view directly for sampling.
    pub external_rt_id: Option<u32>,
    /// Mip levels 1..N, largest first, when they are supplied rather than
    /// derived. Empty means "work them out on upload".
    ///
    /// Block-compressed textures have no choice: a mip chain cannot be box
    /// filtered from BC blocks, and the GPU cannot render into them either, so
    /// the levels have to arrive already encoded. Every sampler in the renderer
    /// asks for trilinear filtering, so a large texture without a chain does
    /// not merely lose quality — it shimmers under any minification at all.
    pub mips: Vec<Arc<Vec<u8>>>,
    /// Stable identity, assigned once at construction and carried by clones.
    ///
    /// The upload cache used to key on the DATA'S ADDRESS, which ties a GPU
    /// texture's identity to the lifetime of its CPU bytes: release them and
    /// the entry is orphaned, free and reallocate and it is silently reused by
    /// somebody else. An id decouples the two, which is what makes
    /// [`Texture::handle`] possible — and that is worth a great deal when the
    /// bytes are gigabytes.
    pub id: u64,
    /// Bumped when [`replace_rgba_bytes`](Self::replace_rgba_bytes) swaps pixel
    /// data so the renderer re-uploads without allocating a new texture id.
    pub upload_seq: u64,
    /// Frames that replace the pixels while the texture stays the same, as
    /// [`VideoTexture`](super::VideoTexture) writes them. The GPU renderers
    /// copy the newest into the texture they already have, and
    /// [`pixels`](Self::pixels) reads it for CPU paths (the path tracer,
    /// exporters). A streamed texture's own `data` may be empty.
    pub stream: Option<Arc<super::TextureStream>>,
}

/// Source of [`Texture::id`]. Monotonic; wrapping it would take 500 years of
/// allocating a texture every nanosecond.
static NEXT_TEXTURE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(crate) fn next_texture_id() -> u64 {
    NEXT_TEXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

const IDENTITY3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

impl Texture {
    /// three.js `Texture.matrix` as the two rows a shader multiplies the UV
    /// by: `uv' = (r0 · (u, v, 1), r1 · (u, v, 1))`.
    ///
    /// With `matrix_auto_update` (the default) this is `Matrix3.setUvTransform(
    /// offset.x, offset.y, repeat.x, repeat.y, rotation, center.x, center.y)`,
    /// which three.js recomputes in `refreshTransformUniform`; otherwise the
    /// user's [`matrix`](Self::matrix).
    pub fn uv_transform_rows(&self) -> [[f32; 3]; 2] {
        if !self.matrix_auto_update {
            let e = &self.matrix;
            // Column-major: element (row r, column c) is e[c * 3 + r].
            return [[e[0], e[3], e[6]], [e[1], e[4], e[7]]];
        }
        let (tx, ty) = (self.offset.x, self.offset.y);
        let (sx, sy) = (self.repeat.x, self.repeat.y);
        let (cx, cy) = (self.center.x, self.center.y);
        let (s, c) = self.rotation.sin_cos();
        [
            [sx * c, sx * s, -sx * (c * cx + s * cy) + cx + tx],
            [-sy * s, sy * c, -sy * (-s * cx + c * cy) + cy + ty],
        ]
    }

    pub fn new(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            format,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            mipmap_filter: Some(TextureFilter::Linear),
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            data: Arc::new(data),
            flip_y: true,
            offset: Vector2::ZERO,
            repeat: Vector2::ONE,
            rotation: 0.0,
            center: Vector2::ZERO,
            channel: 0,
            matrix_auto_update: true,
            matrix: IDENTITY3,
            external_rt_id: None,
            mips: Vec::new(),
            id: next_texture_id(),
            upload_seq: 0,
            stream: None,
        }
    }

    /// Build a Texture that's backed by a RenderTarget's color attachment.
    /// The renderer skips upload and samples directly from the RT view.
    pub fn from_render_target(rt_id: u32, width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            width,
            height,
            format,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            // three.js gives a render target's texture `LinearFilter`: it has
            // no mip chain.
            mipmap_filter: None,
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            data: Arc::new(Vec::new()),
            flip_y: false,
            offset: Vector2::ZERO,
            repeat: Vector2::ONE,
            rotation: 0.0,
            center: Vector2::ZERO,
            channel: 0,
            matrix_auto_update: true,
            matrix: IDENTITY3,
            external_rt_id: Some(rt_id),
            mips: Vec::new(),
            id: next_texture_id(),
            upload_seq: 0,
            stream: None,
        }
    }

    /// Replace level-0 RGBA bytes in place (same width/height). Returns false when
    /// `data` length does not match.
    pub fn replace_rgba_bytes(&mut self, data: Vec<u8>) -> bool {
        let expected = self.format.data_len(self.width, self.height);
        if data.len() != expected {
            return false;
        }
        self.data = Arc::new(data);
        self.upload_seq += 1;
        true
    }

    /// Bytes one pixel occupies. **Panics on block-compressed formats**, which
    /// have no such number.
    ///
    /// Every caller of this in the crate walks the data per pixel — building
    /// mip chains, compositing ice caps, converting height to normals — and not
    /// one of those operations is meaningful on 4x4 blocks. Returning something
    /// plausible instead (4, say, or 0 with a `.max(1)` downstream) would let
    /// all of them run and quietly produce noise. Use
    /// [`TextureFormat::data_len`] for sizes and
    /// [`TextureFormat::is_block_compressed`] to branch.
    /// The pixels as 8-bit RGBA, the right way up for a file.
    ///
    /// `None` when there is nothing to convert: a texture whose bytes live only
    /// on the GPU, or one held block-compressed, which would have to be decoded
    /// first. Rows come out top-first, which is what every image format stores
    /// and what both the glTF and USD writers need — so a texture with
    /// [`flip_y`](Self::flip_y) off, meaning its rows are already bottom-up, is
    /// turned over here rather than being written upside down.
    pub fn to_rgba8(&self) -> Option<Vec<u8>> {
        let count = (self.width as usize).checked_mul(self.height as usize)?;
        if count == 0 || self.data.is_empty() {
            return None;
        }
        let pixels: Vec<u8> = match self.format {
            TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => {
                self.data.get(..count * 4)?.to_vec()
            }
            // One channel becomes grey, so the file says what the channel said
            // however a reader samples it.
            TextureFormat::R8Unorm => self
                .data
                .get(..count)?
                .iter()
                .flat_map(|v| [*v, *v, *v, 255])
                .collect(),
            // Floats down to 8 bits. Anything above 1.0 clamps, which is a
            // real loss and only reachable for HDR or data; one channel
            // becomes grey, as `R8Unorm` does.
            f if f.is_float() => {
                let texels = f.texels_f32(self.data.get(..f.data_len(self.width, self.height))?)?;
                let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                if f.channels() == 1 {
                    texels.iter().flat_map(|v| [byte(*v), byte(*v), byte(*v), 255]).collect()
                } else {
                    texels.iter().map(|v| byte(*v)).collect()
                }
            }
            _ => return None,
        };
        if self.flip_y {
            return Some(pixels);
        }
        let stride = self.width as usize * 4;
        Some(
            pixels
                .chunks_exact(stride)
                .rev()
                .flatten()
                .copied()
                .collect(),
        )
    }

    pub fn bytes_per_pixel(&self) -> usize {
        assert!(
            !self.format.is_block_compressed(),
            "bytes_per_pixel on {:?}: block-compressed textures have no per-pixel \
             stride — use TextureFormat::data_len / is_block_compressed",
            self.format
        );
        self.format.bytes_per_block()
    }

    /// The same pixels and sampler settings, read in a different format.
    ///
    /// For glTF, where one image can legitimately be both: sRGB when it is
    /// baseColour, linear when it is a normal or metallic-roughness map. Gets a
    /// fresh id, because the two are different GPU textures.
    pub fn with_format(&self, format: TextureFormat) -> Self {
        Self {
            format,
            id: next_texture_id(),
            upload_seq: 0,
            mips: Vec::new(),
            ..self.clone()
        }
    }

    /// A twin of this texture with no CPU bytes, for keeping in a material
    /// once the pixels are on the GPU.
    ///
    /// Same [`id`](Self::id), so the renderer resolves it to the same uploaded
    /// texture; empty `data`, so the copy in system memory can be dropped. The
    /// caller must have uploaded it first — see
    /// `HeadlessRenderer::preload_texture` — because nothing here can put the
    /// bytes back.
    ///
    /// This exists for the case where the bytes are large and numerous: a
    /// planet split into tiles holds gigabytes of pixels that are pure
    /// duplication the moment the GPU has them, and on unified memory that
    /// duplication is not even in a different pool.
    pub fn handle(&self) -> Self {
        Self {
            data: Arc::new(Vec::new()),
            mips: Vec::new(),
            ..self.clone()
        }
    }

    /// The pixels as they are now: a [streamed](Self::stream) texture's
    /// newest frame (as RGBA8, converted on this thread if it was written as
    /// YUV), otherwise [`data`](Self::data). Rows in the order `data` holds
    /// them.
    pub fn pixels(&self) -> std::borrow::Cow<'_, [u8]> {
        match &self.stream {
            Some(stream) => std::borrow::Cow::Owned(stream.read(|rgba, _| rgba.to_vec())),
            None => std::borrow::Cow::Borrowed(&self.data[..]),
        }
    }

    /// Whether this is a [`handle`](Self::handle) — metadata with no pixels.
    pub fn is_handle(&self) -> bool {
        self.data.is_empty() && self.width > 0 && self.height > 0
    }

    /// Bytes this texture's level 0 should occupy, block formats included.
    pub fn data_len(&self) -> usize {
        self.format.data_len(self.width, self.height)
    }

    /// A float texture of `values` (`format.channels()` a texel, rows as
    /// `data` holds them), each stored at the format's width: halves rounded
    /// to nearest even, f32s as they are.
    ///
    /// # Panics
    /// If `format` is not a float format or `values` is not a texel's
    /// channels for every one of `width × height`.
    pub fn from_f32(width: u32, height: u32, format: TextureFormat, values: &[f32]) -> Self {
        assert!(format.is_float(), "from_f32 needs a float format, not {format:?}");
        assert_eq!(values.len(), width as usize * height as usize * format.channels(), "values for {width}×{height} {format:?}");
        let data = match format {
            TextureFormat::Rgba16Float | TextureFormat::R16Float => crate::math::float::pack_f16(values),
            _ => crate::math::float::pack_f32(values),
        };
        let mut t = Self::new(width, height, format, data);
        t.mipmap_filter = None;
        if format.needs_float32_filterable() {
            // Sampled nearest unless the device can filter 32-bit floats.
            t.mag_filter = TextureFilter::Nearest;
            t.min_filter = TextureFilter::Nearest;
        }
        t
    }

    /// [`from_f32`](Self::from_f32) of f64s, each rounded once to the
    /// format's width (straight to half, not through f32).
    pub fn from_f64(width: u32, height: u32, format: TextureFormat, values: &[f64]) -> Self {
        match format {
            TextureFormat::Rgba16Float | TextureFormat::R16Float => {
                assert_eq!(values.len(), width as usize * height as usize * format.channels(), "values for {width}×{height} {format:?}");
                let mut t = Self::new(width, height, format, crate::math::float::pack_f16_from_f64(values));
                t.mipmap_filter = None;
                t
            }
            _ => Self::from_f32(width, height, format, &values.iter().map(|v| *v as f32).collect::<Vec<_>>()),
        }
    }

    /// Level 0's texels as f32s, every channel, if the bytes are here and
    /// uncompressed ([`TextureFormat::texels_f32`]).
    pub fn texels_f32(&self) -> Option<Vec<f32>> {
        self.format.texels_f32(self.data.get(..self.data_len())?)
    }

    /// Build a solid-color 1x1 texture in linear RGBA8. Useful as a placeholder
    /// when a material slot is unused.
    pub fn solid(rgba: [u8; 4], format: TextureFormat) -> Self {
        Self::new(1, 1, format, rgba.to_vec())
    }
}

