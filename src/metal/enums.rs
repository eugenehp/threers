//! Metal enumerants and by-value structs.
//!
//! Metal's headers are C enums and small structs; without a binding generator
//! they have to be restated. The values are ABI — they are what the framework
//! reads — so they are transcribed from `<Metal/Metal.h>` and must not be
//! renumbered for convenience.
//!
//! Every struct here is `#[repr(C)]` and is passed to `objc_msgSend` by value,
//! which is correct on both arm64 and x86_64 because rustc lays `repr(C)` out
//! with the platform C ABI.

#![allow(dead_code)]

/// `NSUInteger` — the width of a pointer on every platform Metal runs on.
pub type NSUInteger = usize;

/// `MTLPixelFormat`.
pub mod pixel_format {
    use super::NSUInteger;
    pub const R8_UNORM: NSUInteger = 10;
    pub const RG8_UNORM: NSUInteger = 30;
    pub const RGBA8_UNORM: NSUInteger = 70;
    pub const RGBA8_UNORM_SRGB: NSUInteger = 71;
    pub const BGRA8_UNORM: NSUInteger = 80;
    pub const BGRA8_UNORM_SRGB: NSUInteger = 81;
    pub const RGBA16_FLOAT: NSUInteger = 115;
    pub const RGBA32_FLOAT: NSUInteger = 125;
    /// BC1 ("DXT1"), sRGB. Desktop-class GPUs only — see
    /// `MTLDevice.supportsBCTextureCompression`.
    pub const BC1_RGBA_SRGB: NSUInteger = 131;
    /// BC7, sRGB. Same availability caveat as BC1.
    pub const BC7_RGBA_SRGB: NSUInteger = 145;
    pub const DEPTH32_FLOAT: NSUInteger = 252;
    pub const DEPTH32_FLOAT_STENCIL8: NSUInteger = 260;

    /// Bytes per pixel for the formats this backend uploads or reads back.
    pub fn bytes_per_pixel(format: NSUInteger) -> Option<usize> {
        Some(match format {
            R8_UNORM => 1,
            RG8_UNORM => 2,
            RGBA8_UNORM | RGBA8_UNORM_SRGB | BGRA8_UNORM | BGRA8_UNORM_SRGB => 4,
            RGBA16_FLOAT => 8,
            RGBA32_FLOAT => 16,
            _ => return None,
        })
    }

    /// Whether the GPU applies the sRGB transfer function on write/read.
    pub fn is_srgb(format: NSUInteger) -> bool {
        matches!(format, RGBA8_UNORM_SRGB | BGRA8_UNORM_SRGB)
    }

    /// Whether channel order is BGRA rather than RGBA (the `CAMetalLayer` default).
    pub fn is_bgra(format: NSUInteger) -> bool {
        matches!(format, BGRA8_UNORM | BGRA8_UNORM_SRGB)
    }
}

/// `MTLTextureType`.
pub mod texture_type {
    use super::NSUInteger;
    pub const TYPE_2D: NSUInteger = 2;
    pub const TYPE_2D_ARRAY: NSUInteger = 3;
    pub const TYPE_2D_MULTISAMPLE: NSUInteger = 4;
    pub const TYPE_CUBE: NSUInteger = 5;
}

/// `MTLTextureUsage` — a bit mask.
pub mod texture_usage {
    use super::NSUInteger;
    pub const UNKNOWN: NSUInteger = 0;
    pub const SHADER_READ: NSUInteger = 1;
    pub const SHADER_WRITE: NSUInteger = 2;
    pub const RENDER_TARGET: NSUInteger = 4;
}

/// `MTLStorageMode`, as it appears on a texture descriptor.
pub mod storage_mode {
    use super::NSUInteger;
    pub const SHARED: NSUInteger = 0;
    pub const MANAGED: NSUInteger = 1;
    pub const PRIVATE: NSUInteger = 2;
    pub const MEMORYLESS: NSUInteger = 3;
}

/// `MTLResourceOptions` — storage mode shifted into the resource-option bits.
pub mod resource_options {
    use super::NSUInteger;
    pub const STORAGE_MODE_SHARED: NSUInteger = 0 << 4;
    pub const STORAGE_MODE_MANAGED: NSUInteger = 1 << 4;
    pub const STORAGE_MODE_PRIVATE: NSUInteger = 2 << 4;
}

/// `MTLLoadAction`.
pub mod load_action {
    use super::NSUInteger;
    pub const DONT_CARE: NSUInteger = 0;
    pub const LOAD: NSUInteger = 1;
    pub const CLEAR: NSUInteger = 2;
}

/// `MTLStoreAction`.
pub mod store_action {
    use super::NSUInteger;
    pub const DONT_CARE: NSUInteger = 0;
    pub const STORE: NSUInteger = 1;
    pub const MULTISAMPLE_RESOLVE: NSUInteger = 2;
    pub const STORE_AND_MULTISAMPLE_RESOLVE: NSUInteger = 3;
}

/// `MTLPrimitiveType`.
pub mod primitive {
    use super::NSUInteger;
    pub const POINT: NSUInteger = 0;
    pub const LINE: NSUInteger = 1;
    pub const LINE_STRIP: NSUInteger = 2;
    pub const TRIANGLE: NSUInteger = 3;
    pub const TRIANGLE_STRIP: NSUInteger = 4;
}

/// `MTLPrimitiveTopologyClass`.
///
/// A pipeline whose vertex function writes `render_target_array_index` must
/// declare which of these it rasterises — Metal rejects it otherwise.
pub mod topology_class {
    use super::NSUInteger;
    pub const UNSPECIFIED: NSUInteger = 0;
    pub const POINT: NSUInteger = 1;
    pub const LINE: NSUInteger = 2;
    pub const TRIANGLE: NSUInteger = 3;

    /// The class a `MTLPrimitiveType` belongs to.
    pub fn of(primitive: NSUInteger) -> NSUInteger {
        match primitive {
            super::primitive::POINT => POINT,
            super::primitive::LINE | super::primitive::LINE_STRIP => LINE,
            _ => TRIANGLE,
        }
    }
}

/// `MTLIndexType`.
pub mod index_type {
    use super::NSUInteger;
    pub const UINT16: NSUInteger = 0;
    pub const UINT32: NSUInteger = 1;
}

/// `MTLCompareFunction`.
pub mod compare {
    use super::NSUInteger;
    pub const NEVER: NSUInteger = 0;
    pub const LESS: NSUInteger = 1;
    pub const EQUAL: NSUInteger = 2;
    pub const LESS_EQUAL: NSUInteger = 3;
    pub const GREATER: NSUInteger = 4;
    pub const NOT_EQUAL: NSUInteger = 5;
    pub const GREATER_EQUAL: NSUInteger = 6;
    pub const ALWAYS: NSUInteger = 7;
}

/// `MTLCullMode`.
pub mod cull {
    use super::NSUInteger;
    pub const NONE: NSUInteger = 0;
    pub const FRONT: NSUInteger = 1;
    pub const BACK: NSUInteger = 2;
}

/// `MTLWinding`.
pub mod winding {
    use super::NSUInteger;
    pub const CLOCKWISE: NSUInteger = 0;
    pub const COUNTER_CLOCKWISE: NSUInteger = 1;
}

/// `MTLTriangleFillMode`.
pub mod fill_mode {
    use super::NSUInteger;
    pub const FILL: NSUInteger = 0;
    pub const LINES: NSUInteger = 1;
}

/// `MTLBlendFactor`.
pub mod blend_factor {
    use super::NSUInteger;
    pub const ZERO: NSUInteger = 0;
    pub const ONE: NSUInteger = 1;
    pub const SOURCE_ALPHA: NSUInteger = 4;
    pub const ONE_MINUS_SOURCE_ALPHA: NSUInteger = 5;
}

/// `MTLBlendOperation`.
pub mod blend_op {
    use super::NSUInteger;
    pub const ADD: NSUInteger = 0;
}

/// `MTLSamplerMinMagFilter` / `MTLSamplerMipFilter`.
pub mod filter {
    use super::NSUInteger;
    pub const NEAREST: NSUInteger = 0;
    pub const LINEAR: NSUInteger = 1;
    pub const MIP_NOT_MIPMAPPED: NSUInteger = 0;
    pub const MIP_NEAREST: NSUInteger = 1;
    pub const MIP_LINEAR: NSUInteger = 2;
}

/// `MTLSamplerAddressMode`.
pub mod address_mode {
    use super::NSUInteger;
    pub const CLAMP_TO_EDGE: NSUInteger = 0;
    pub const MIRROR_CLAMP_TO_EDGE: NSUInteger = 1;
    pub const REPEAT: NSUInteger = 2;
    pub const MIRROR_REPEAT: NSUInteger = 3;
}

/// `MTLOrigin`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MTLOrigin {
    pub x: NSUInteger,
    pub y: NSUInteger,
    pub z: NSUInteger,
}

/// `MTLSize`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MTLSize {
    pub width: NSUInteger,
    pub height: NSUInteger,
    pub depth: NSUInteger,
}

/// `MTLRegion`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

impl MTLRegion {
    /// The whole of a 2D image.
    pub fn image_2d(width: u32, height: u32) -> Self {
        Self {
            origin: MTLOrigin::default(),
            size: MTLSize {
                width: width as NSUInteger,
                height: height as NSUInteger,
                depth: 1,
            },
        }
    }
}

/// `MTLClearColor` — four `double`s, always linear (never sRGB-encoded, even
/// for an sRGB attachment: the GPU applies the transfer function on write).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MTLClearColor {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
    pub alpha: f64,
}

/// `MTLViewport`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct MTLViewport {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
    pub znear: f64,
    pub zfar: f64,
}

/// `NSRange`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NSRange {
    pub location: NSUInteger,
    pub length: NSUInteger,
}

/// `CGSize`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct CGSize {
    pub width: f64,
    pub height: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_metal() {
        // These sizes are the ABI contract with the framework; a mismatch is a
        // silently corrupt argument list, not a compile error.
        let word = std::mem::size_of::<usize>();
        assert_eq!(std::mem::size_of::<MTLOrigin>(), 3 * word);
        assert_eq!(std::mem::size_of::<MTLSize>(), 3 * word);
        assert_eq!(std::mem::size_of::<MTLRegion>(), 6 * word);
        assert_eq!(std::mem::size_of::<MTLClearColor>(), 32);
        assert_eq!(std::mem::size_of::<MTLViewport>(), 48);
        assert_eq!(std::mem::size_of::<NSRange>(), 2 * word);
        assert_eq!(std::mem::size_of::<CGSize>(), 16);
    }

    #[test]
    fn pixel_format_metadata() {
        assert_eq!(
            pixel_format::bytes_per_pixel(pixel_format::RGBA8_UNORM_SRGB),
            Some(4)
        );
        assert_eq!(
            pixel_format::bytes_per_pixel(pixel_format::RGBA16_FLOAT),
            Some(8)
        );
        assert_eq!(
            pixel_format::bytes_per_pixel(pixel_format::DEPTH32_FLOAT),
            None
        );
        assert!(pixel_format::is_srgb(pixel_format::BGRA8_UNORM_SRGB));
        assert!(!pixel_format::is_srgb(pixel_format::RGBA8_UNORM));
        assert!(pixel_format::is_bgra(pixel_format::BGRA8_UNORM));
        assert!(!pixel_format::is_bgra(pixel_format::RGBA8_UNORM_SRGB));
    }
}
