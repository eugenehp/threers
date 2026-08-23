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
}

/// Source of [`Texture::id`]. Monotonic; wrapping it would take 500 years of
/// allocating a texture every nanosecond.
static NEXT_TEXTURE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_texture_id() -> u64 {
    NEXT_TEXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Texture {
    pub fn new(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            format,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            data: Arc::new(data),
            flip_y: true,
            offset: Vector2::ZERO,
            repeat: Vector2::ONE,
            rotation: 0.0,
            external_rt_id: None,
            mips: Vec::new(),
            id: next_texture_id(),
            upload_seq: 0,
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
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            data: Arc::new(Vec::new()),
            flip_y: false,
            offset: Vector2::ZERO,
            repeat: Vector2::ONE,
            rotation: 0.0,
            external_rt_id: Some(rt_id),
            mips: Vec::new(),
            id: next_texture_id(),
            upload_seq: 0,
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
    pub fn bytes_per_pixel(&self) -> usize {
        assert!(
            !self.format.is_block_compressed(),
            "bytes_per_pixel on {:?}: block-compressed textures have no per-pixel \
             stride — use TextureFormat::data_len / is_block_compressed",
            self.format
        );
        self.format.bytes_per_block()
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

    /// Whether this is a [`handle`](Self::handle) — metadata with no pixels.
    pub fn is_handle(&self) -> bool {
        self.data.is_empty() && self.width > 0 && self.height > 0
    }

    /// Bytes this texture's level 0 should occupy, block formats included.
    pub fn data_len(&self) -> usize {
        self.format.data_len(self.width, self.height)
    }

    /// Build a solid-color 1x1 texture in linear RGBA8. Useful as a placeholder
    /// when a material slot is unused.
    pub fn solid(rgba: [u8; 4], format: TextureFormat) -> Self {
        Self::new(1, 1, format, rgba.to_vec())
    }
}
