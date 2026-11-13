use crate::math::Vector2;
use std::sync::Arc;

/// The ASTC block footprints: pixels a 128-bit block covers, from 4x4 (8 bits
/// a pixel) to 12x12 (0.89).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstcBlock {
    B4x4,
    B5x4,
    B5x5,
    B6x5,
    B6x6,
    B8x5,
    B8x6,
    B8x8,
    B10x5,
    B10x6,
    B10x8,
    B10x10,
    B12x10,
    B12x12,
}

impl AstcBlock {
    /// Every footprint, smallest first.
    pub const ALL: [AstcBlock; 14] = [
        Self::B4x4,
        Self::B5x4,
        Self::B5x5,
        Self::B6x5,
        Self::B6x6,
        Self::B8x5,
        Self::B8x6,
        Self::B8x8,
        Self::B10x5,
        Self::B10x6,
        Self::B10x8,
        Self::B10x10,
        Self::B12x10,
        Self::B12x12,
    ];

    /// `(width, height)` in pixels.
    pub fn dims(self) -> (u32, u32) {
        match self {
            Self::B4x4 => (4, 4),
            Self::B5x4 => (5, 4),
            Self::B5x5 => (5, 5),
            Self::B6x5 => (6, 5),
            Self::B6x6 => (6, 6),
            Self::B8x5 => (8, 5),
            Self::B8x6 => (8, 6),
            Self::B8x8 => (8, 8),
            Self::B10x5 => (10, 5),
            Self::B10x6 => (10, 6),
            Self::B10x8 => (10, 8),
            Self::B10x10 => (10, 10),
            Self::B12x10 => (12, 10),
            Self::B12x12 => (12, 12),
        }
    }

    /// The footprint with these dimensions, if ASTC has one.
    pub fn from_dims(width: u32, height: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.dims() == (width, height))
    }
}

/// Which family of block compression a format belongs to. A GPU supports a
/// family as a whole or not at all: BC on desktops, ETC2 and ASTC on phones
/// and headsets, all three on Apple silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    Bc,
    Etc2,
    Astc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// Standard 8-bit sRGB color (e.g. baseColor / albedo).
    Rgba8UnormSrgb,
    /// Linear 8-bit (e.g. roughness, metalness, normal maps).
    Rgba8Unorm,
    /// Single-channel 8-bit.
    R8Unorm,
    /// Two 8-bit channels (2 bytes/px): a two-channel normal map's X and Y, or
    /// any pair of quantities.
    Rg8Unorm,
    /// One 16-bit unsigned normalized channel (2 bytes/px): what a 16-bit
    /// greyscale PNG holds, such as a height map, at 65536 levels rather than
    /// 256. Native where the GPU has 16-bit norm formats; elsewhere uploaded as
    /// 32-bit float where that can be filtered, else as 16-bit float, which
    /// keeps 11 bits.
    R16Unorm,
    /// Four 16-bit unsigned normalized channels (8 bytes/px): a 16-bit RGBA
    /// PNG. Uploaded like [`R16Unorm`](Self::R16Unorm) where the GPU lacks it.
    Rgba16Unorm,
    /// 10 bits each for R, G and B and 2 for alpha, packed in a little-endian
    /// u32 (R in the low bits): four times the levels of 8-bit colour in the
    /// same 4 bytes, for linear data that bands at 8 bits.
    Rgb10a2Unorm,
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
    /// Two 16-bit float channels (4 bytes/px): a flow or velocity field.
    Rg16Float,
    /// One 32-bit float channel (4 bytes/px): data that must keep f32
    /// precision — heights, distances, ids up to 2^24. Filtering it needs
    /// `wgpu::Features::FLOAT32_FILTERABLE`; without it, sample it nearest.
    R32Float,
    /// Two 32-bit float channels (8 bytes/px). Filtering needs
    /// `FLOAT32_FILTERABLE`, as for [`R32Float`](Self::R32Float).
    Rg32Float,
    /// Four 32-bit float channels (16 bytes/px): positions, or HDR colour
    /// that must not round to half. Filterable as `R32Float`.
    Rgba32Float,
    /// Three unsigned floats in a little-endian u32: 11 bits for R and G, 10
    /// for B (6 bits of mantissa, 5 of exponent each, no sign). HDR colour in
    /// half the bytes of `Rgba16Float`, opaque.
    Rg11b10Ufloat,
    /// Three 9-bit mantissas sharing one 5-bit exponent, in a little-endian
    /// u32: HDR colour at 4 bytes a pixel, filterable everywhere, which makes it
    /// the compact format for environment maps. Opaque; channels much darker
    /// than the brightest lose precision, as they share its exponent.
    Rgb9e5Ufloat,
    /// or 2.32 with a full mip chain.
    Bc1RgbaUnormSrgb,
    /// BC1, linear: for data rather than colour. Same blocks as
    /// [`Bc1RgbaUnormSrgb`](Self::Bc1RgbaUnormSrgb), read without the sRGB
    /// curve.
    Bc1RgbaUnorm,
    /// BC3 ("DXT5"), sRGB. 16 bytes a block: BC1 colour (always four colours)
    /// plus a BC4 block for alpha, so alpha is smooth rather than on or off.
    Bc3RgbaUnormSrgb,
    /// BC3, linear.
    Bc3RgbaUnorm,
    /// BC4: one channel, 8 bytes a block — half a byte a pixel. Two 8-bit
    /// endpoints and eight steps between them, for roughness, occlusion or
    /// height.
    Bc4RUnorm,
    /// BC5: two BC4 channels, 16 bytes a block — the format for tangent-space
    /// normal maps (X and Y; Z is rebuilt in the shader).
    Bc5RgUnorm,
    /// BC6H, unsigned: HDR RGB at one byte a pixel, an eighth of
    /// `Rgba16Float`. For environment maps and skies. Opaque. Decodes to
    /// `Rgba16Float` where the GPU has no BC.
    Bc6hRgbUfloat,
    /// BC6H, signed: as [`Bc6hRgbUfloat`](Self::Bc6hRgbUfloat), with negative
    /// values.
    Bc6hRgbFloat,
    /// BC7, sRGB. 4x4 blocks in 16 bytes — one byte a pixel, a quarter of
    /// `Rgba8UnormSrgb`.
    ///
    /// Eight block modes with variable partitioning and endpoint precision, so
    /// it holds gradients that BC1 bands. No encoder here — supply the blocks —
    /// but the format, upload path and mip handling all work, so an externally
    /// encoded map can be used as-is.
    Bc7RgbaUnormSrgb,
    /// BC7, linear: normal, roughness and occlusion maps at BC7 quality.
    Bc7RgbaUnorm,
    /// ETC2 RGB, sRGB: 8 bytes a 4x4 block, opaque. The compressed format every
    /// OpenGL ES 3 / WebGL2-with-ETC phone decodes in hardware.
    Etc2Rgb8UnormSrgb,
    /// ETC2 RGB, linear.
    Etc2Rgb8Unorm,
    /// ETC2 RGBA (ETC2 colour plus an EAC alpha block), sRGB: 16 bytes a block.
    Etc2Rgba8UnormSrgb,
    /// ETC2 RGBA, linear.
    Etc2Rgba8Unorm,
    /// ASTC (LDR profile): 16 bytes a block at any [`AstcBlock`] footprint, so
    /// one format spans 8 bits a pixel (4x4) down to 0.89 (12x12). What iOS,
    /// visionOS and Android hardware decode natively. HDR blocks decode to the
    /// error colour, as the LDR profile requires.
    Astc { block: AstcBlock, srgb: bool },
}

impl TextureFormat {
    /// Every format: the ASTC ones at each footprint, sRGB and linear.
    pub fn all() -> Vec<TextureFormat> {
        use TextureFormat::*;
        let mut v = vec![
            Rgba8UnormSrgb,
            Rgba8Unorm,
            R8Unorm,
            Rg8Unorm,
            R16Unorm,
            Rgba16Unorm,
            Rgb10a2Unorm,
            Rgba16Float,
            R16Float,
            Rg16Float,
            R32Float,
            Rg32Float,
            Rgba32Float,
            Rg11b10Ufloat,
            Rgb9e5Ufloat,
            Bc1RgbaUnormSrgb,
            Bc1RgbaUnorm,
            Bc3RgbaUnormSrgb,
            Bc3RgbaUnorm,
            Bc4RUnorm,
            Bc5RgUnorm,
            Bc6hRgbUfloat,
            Bc6hRgbFloat,
            Bc7RgbaUnormSrgb,
            Bc7RgbaUnorm,
            Etc2Rgb8UnormSrgb,
            Etc2Rgb8Unorm,
            Etc2Rgba8UnormSrgb,
            Etc2Rgba8Unorm,
        ];
        for block in AstcBlock::ALL {
            v.push(Astc { block, srgb: true });
            v.push(Astc { block, srgb: false });
        }
        v
    }

    /// A number for this format that never changes between versions: cache
    /// keys and the `.tex` blob both store it. New formats get new codes.
    pub fn code(self) -> u16 {
        use TextureFormat::*;
        match self {
            Rgba8Unorm => 0,
            Rgba8UnormSrgb => 1,
            R8Unorm => 2,
            Rgba16Float => 3,
            Bc1RgbaUnormSrgb => 4,
            Bc7RgbaUnormSrgb => 5,
            R16Float => 6,
            R32Float => 7,
            Rgba32Float => 8,
            Rg8Unorm => 9,
            R16Unorm => 10,
            Rgba16Unorm => 11,
            Rgb10a2Unorm => 12,
            Rg16Float => 13,
            Rg32Float => 14,
            Rg11b10Ufloat => 15,
            Rgb9e5Ufloat => 16,
            Bc1RgbaUnorm => 17,
            Bc3RgbaUnormSrgb => 18,
            Bc3RgbaUnorm => 19,
            Bc4RUnorm => 20,
            Bc5RgUnorm => 21,
            Bc6hRgbUfloat => 22,
            Bc6hRgbFloat => 23,
            Bc7RgbaUnorm => 24,
            Etc2Rgb8UnormSrgb => 25,
            Etc2Rgb8Unorm => 26,
            Etc2Rgba8UnormSrgb => 27,
            Etc2Rgba8Unorm => 28,
            // 0x100 | footprint << 1 | srgb.
            Astc { block, srgb } => {
                let i = AstcBlock::ALL.iter().position(|b| *b == block).unwrap() as u16;
                0x100 | (i << 1) | srgb as u16
            }
        }
    }

    /// The format with this [`code`](Self::code).
    pub fn from_code(code: u16) -> Option<Self> {
        Self::all().into_iter().find(|f| f.code() == code)
    }

    /// The block-compression family, `None` for uncompressed formats.
    pub fn compression(self) -> Option<Compression> {
        use TextureFormat::*;
        match self {
            Bc1RgbaUnormSrgb | Bc1RgbaUnorm | Bc3RgbaUnormSrgb | Bc3RgbaUnorm | Bc4RUnorm
            | Bc5RgUnorm | Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnormSrgb | Bc7RgbaUnorm => {
                Some(Compression::Bc)
            }
            Etc2Rgb8UnormSrgb | Etc2Rgb8Unorm | Etc2Rgba8UnormSrgb | Etc2Rgba8Unorm => {
                Some(Compression::Etc2)
            }
            Astc { .. } => Some(Compression::Astc),
            _ => None,
        }
    }

    /// Block-compressed formats store a `w x h` block of pixels as one fixed
    /// unit and cannot be addressed per pixel at all.
    pub fn is_block_compressed(self) -> bool {
        self.compression().is_some()
    }

    /// Whether the GPU decodes texels through the sRGB curve.
    pub fn is_srgb(self) -> bool {
        use TextureFormat::*;
        match self {
            Rgba8UnormSrgb | Bc1RgbaUnormSrgb | Bc3RgbaUnormSrgb | Bc7RgbaUnormSrgb
            | Etc2Rgb8UnormSrgb | Etc2Rgba8UnormSrgb => true,
            Astc { srgb, .. } => srgb,
            _ => false,
        }
    }

    /// The same bytes read with (`srgb`) or without the sRGB curve: three.js's
    /// `colorSpace` for this format. `None` when the format has no such twin
    /// (floats, and one- and two-channel data).
    pub fn with_srgb(self, srgb: bool) -> Option<Self> {
        use TextureFormat::*;
        let pair = match self {
            Rgba8UnormSrgb | Rgba8Unorm => (Rgba8UnormSrgb, Rgba8Unorm),
            Bc1RgbaUnormSrgb | Bc1RgbaUnorm => (Bc1RgbaUnormSrgb, Bc1RgbaUnorm),
            Bc3RgbaUnormSrgb | Bc3RgbaUnorm => (Bc3RgbaUnormSrgb, Bc3RgbaUnorm),
            Bc7RgbaUnormSrgb | Bc7RgbaUnorm => (Bc7RgbaUnormSrgb, Bc7RgbaUnorm),
            Etc2Rgb8UnormSrgb | Etc2Rgb8Unorm => (Etc2Rgb8UnormSrgb, Etc2Rgb8Unorm),
            Etc2Rgba8UnormSrgb | Etc2Rgba8Unorm => (Etc2Rgba8UnormSrgb, Etc2Rgba8Unorm),
            Astc { block, .. } => return Some(Astc { block, srgb }),
            _ => return None,
        };
        Some(if srgb { pair.0 } else { pair.1 })
    }

    /// Pixels per block, `(1, 1)` for uncompressed formats.
    pub fn block_dim(self) -> (u32, u32) {
        match self {
            Self::Astc { block, .. } => block.dims(),
            f if f.is_block_compressed() => (4, 4),
            _ => (1, 1),
        }
    }

    /// Bytes per block — per pixel, for the uncompressed formats.
    pub fn bytes_per_block(self) -> usize {
        use TextureFormat::*;
        match self {
            R8Unorm => 1,
            Rg8Unorm | R16Unorm | R16Float => 2,
            Rgba8UnormSrgb | Rgba8Unorm | Rgb10a2Unorm | Rg16Float | R32Float | Rg11b10Ufloat
            | Rgb9e5Ufloat => 4,
            Rgba16Unorm | Rgba16Float | Rg32Float => 8,
            Rgba32Float => 16,
            Bc1RgbaUnormSrgb | Bc1RgbaUnorm | Bc4RUnorm | Etc2Rgb8UnormSrgb | Etc2Rgb8Unorm => 8,
            Bc3RgbaUnormSrgb
            | Bc3RgbaUnorm
            | Bc5RgUnorm
            | Bc6hRgbUfloat
            | Bc6hRgbFloat
            | Bc7RgbaUnormSrgb
            | Bc7RgbaUnorm
            | Etc2Rgba8UnormSrgb
            | Etc2Rgba8Unorm
            | Astc { .. } => 16,
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

    /// Whether texels are floats (any width, packed ones included), which hold
    /// values outside `[0, 1]` and are not colour-encoded.
    pub fn is_float(self) -> bool {
        use TextureFormat::*;
        matches!(
            self,
            Rgba16Float
                | R16Float
                | Rg16Float
                | R32Float
                | Rg32Float
                | Rgba32Float
                | Rg11b10Ufloat
                | Rgb9e5Ufloat
                | Bc6hRgbUfloat
                | Bc6hRgbFloat
        )
    }

    /// Channels a texel holds. The packed three-channel formats and BC6H
    /// count as four, alpha reading 1, as the GPU samples them.
    pub fn channels(self) -> usize {
        use TextureFormat::*;
        match self {
            R8Unorm | R16Unorm | R16Float | R32Float | Bc4RUnorm => 1,
            Rg8Unorm | Rg16Float | Rg32Float | Bc5RgUnorm => 2,
            _ => 4,
        }
    }

    /// Whether linear filtering needs `wgpu::Features::FLOAT32_FILTERABLE`
    /// (32-bit floats; WebGL2 needs `OES_texture_float_linear` for it).
    pub fn needs_float32_filterable(self) -> bool {
        matches!(self, Self::R32Float | Self::Rg32Float | Self::Rgba32Float)
    }

    /// Whether the GPU needs `wgpu::Features::TEXTURE_FORMAT_16BIT_NORM`.
    pub fn needs_16bit_norm(self) -> bool {
        matches!(self, Self::R16Unorm | Self::Rgba16Unorm)
    }

    /// Texels of uncompressed `data` in this format as f32s, every channel
    /// (the unorm formats as `value / max`, sRGB not decoded; the packed
    /// three-channel ones as RGBA with alpha 1). `None` for the block formats:
    /// [`decode`](crate::textures::codec::decode) them first.
    pub fn texels_f32(self, data: &[u8]) -> Option<Vec<f32>> {
        use crate::textures::codec::packed;
        use TextureFormat::*;
        Some(match self {
            Rgba8UnormSrgb | Rgba8Unorm | R8Unorm | Rg8Unorm => {
                data.iter().map(|b| *b as f32 / 255.0).collect()
            }
            R16Unorm | Rgba16Unorm => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b) as f32 / 65535.0)
                .collect(),
            Rgb10a2Unorm => packed::words(data)
                .flat_map(packed::unpack_rgb10a2)
                .collect(),
            Rg11b10Ufloat => packed::words(data)
                .flat_map(packed::unpack_rg11b10)
                .collect(),
            Rgb9e5Ufloat => packed::words(data)
                .flat_map(packed::unpack_rgb9e5)
                .collect(),
            Rgba16Float | R16Float | Rg16Float => crate::math::float::unpack_f16(data),
            R32Float | Rg32Float | Rgba32Float => crate::math::float::unpack_f32(data),
            _ => return None,
        })
    }

    /// Pack f32 texels (in [`channels`](Self::channels) order) into this
    /// uncompressed format, rounding each to the nearest value it can hold.
    /// `None` for the block formats, which need an encoder.
    pub fn pack_f32(self, texels: &[f32]) -> Option<Vec<u8>> {
        use crate::textures::codec::packed;
        use TextureFormat::*;
        let unorm8 = |v: &f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        Some(match self {
            Rgba8UnormSrgb | Rgba8Unorm | R8Unorm | Rg8Unorm => texels.iter().map(unorm8).collect(),
            R16Unorm | Rgba16Unorm => texels
                .iter()
                .flat_map(|v| ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes())
                .collect(),
            Rgb10a2Unorm => texels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| packed::pack_rgb10a2(*c).to_le_bytes())
                .collect(),
            Rg11b10Ufloat => texels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| packed::pack_rg11b10([c[0], c[1], c[2]]).to_le_bytes())
                .collect(),
            Rgb9e5Ufloat => texels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| packed::pack_rgb9e5([c[0], c[1], c[2]]).to_le_bytes())
                .collect(),
            Rgba16Float | R16Float | Rg16Float => crate::math::float::pack_f16(texels),
            R32Float | Rg32Float | Rgba32Float => crate::math::float::pack_f32(texels),
            _ => return None,
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
        out.extend_from_slice(&crate::math::float::f32_to_f16(*v).to_le_bytes());
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
    ///
    /// A render target's texture is read upside down after that: its first
    /// row is the top of the picture, where a material's `v` runs up from
    /// the bottom, so without it a camera's picture on a plane stood on its
    /// head. (three.js's WebGL targets are stored bottom row first, and need
    /// nothing.)
    pub fn uv_transform_rows(&self) -> [[f32; 3]; 2] {
        let [r0, r1] = self.user_uv_rows();
        if self.external_rt_id.is_some() {
            return [r0, [-r1[0], -r1[1], 1.0 - r1[2]]];
        }
        [r0, r1]
    }

    /// [`uv_transform_rows`](Self::uv_transform_rows) as the texture's own
    /// settings give it.
    fn user_uv_rows(&self) -> [[f32; 3]; 2] {
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
                    texels
                        .iter()
                        .flat_map(|v| [byte(*v), byte(*v), byte(*v), 255])
                        .collect()
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
    /// `Renderer::preload_texture` — because nothing here can put the
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
        assert!(
            format.is_float(),
            "from_f32 needs a float format, not {format:?}"
        );
        assert_eq!(
            values.len(),
            width as usize * height as usize * format.channels(),
            "values for {width}×{height} {format:?}"
        );
        let data = match format {
            TextureFormat::Rgba16Float | TextureFormat::R16Float => {
                crate::math::float::pack_f16(values)
            }
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
                assert_eq!(
                    values.len(),
                    width as usize * height as usize * format.channels(),
                    "values for {width}×{height} {format:?}"
                );
                let mut t = Self::new(
                    width,
                    height,
                    format,
                    crate::math::float::pack_f16_from_f64(values),
                );
                t.mipmap_filter = None;
                t
            }
            _ => Self::from_f32(
                width,
                height,
                format,
                &values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            ),
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

#[cfg(test)]
mod uv_tests {
    use super::*;

    /// A render target's texture is read upside down (its first row is the
    /// picture's top), after the texture's own transform.
    #[test]
    fn render_target_textures_are_read_upright() {
        let plain = Texture::new(4, 4, TextureFormat::Rgba8Unorm, vec![0; 64]);
        assert_eq!(plain.uv_transform_rows(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let mut rt = Texture::from_render_target(7, 4, 4, TextureFormat::Rgba8Unorm);
        assert_eq!(rt.uv_transform_rows(), [[1.0, 0.0, 0.0], [-0.0, -1.0, 1.0]]);
        // Its own offset first: v' = 1 - (v + 0.25).
        rt.offset = Vector2::new(0.0, 0.25);
        let [_, r1] = rt.uv_transform_rows();
        assert_eq!(r1[1] * 0.5 + r1[2], 1.0 - 0.75);
    }
}
