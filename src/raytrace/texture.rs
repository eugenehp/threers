//! CPU-side texture sampling for the path tracer.
//!
//! The raster renderer hands textures to the GPU and lets a sampler unit do
//! this. There is no sampler unit here, so a [`Texture`] is decoded once into a
//! [`CpuTexture`] — block formats expanded, sRGB flagged, half floats widened —
//! and every shading point reads from that.
//!
//! Decoding once is the whole point: a BC1 map decoded per sample would cost
//! more than the ray that found it, and an 8-bit map converted from sRGB with
//! `powf` per channel per sample is not much better. Blocks are expanded at
//! build time and sRGB goes through a 256-entry table.

use crate::math::{Vector2, Vector3};
use crate::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Decoded texels. LDR stays 8-bit — four bytes a pixel rather than sixteen,
/// which for a 4K albedo map is the difference between 67 MB and 268.
#[derive(Debug, Clone)]
enum Texels {
    /// RGBA8. `srgb` says whether the stored values need decoding on read.
    Ldr { data: Vec<u8>, srgb: bool },
    /// RGBA32F, already linear. This is where an HDR environment lives.
    Hdr(Vec<f32>),
}

/// A texture in the form the tracer samples: decoded, linear on read, with the
/// three.js sampler state (wrap, offset/repeat/rotation, flip) applied here
/// rather than in a shader.
#[derive(Debug, Clone)]
pub struct CpuTexture {
    width: u32,
    height: u32,
    texels: Texels,
    wrap_s: TextureWrap,
    wrap_t: TextureWrap,
    bilinear: bool,
    flip_y: bool,
    /// Whether any texel is less than fully opaque. Decided once here, because
    /// the alternative is deciding it per shadow ray.
    has_alpha: bool,
    offset: Vector2,
    repeat: Vector2,
    rotation: f32,
}

impl CpuTexture {
    /// Decode a [`Texture`]. Returns `None` for a texture with no CPU-side
    /// pixels at all — a render-target handle, or a format without a decoder
    /// (BC7, which the crate can upload but not expand).
    pub fn from_texture(tex: &Texture) -> Option<Self> {
        if tex.is_handle() || tex.width == 0 || tex.height == 0 {
            return None;
        }
        let (w, h) = (tex.width, tex.height);
        let src = tex.data.as_ref();
        let texels = match tex.format {
            TextureFormat::Rgba8UnormSrgb => Texels::Ldr {
                data: fit(src, (w * h * 4) as usize),
                srgb: true,
            },
            TextureFormat::Rgba8Unorm => Texels::Ldr {
                data: fit(src, (w * h * 4) as usize),
                srgb: false,
            },
            TextureFormat::R8Unorm => {
                // Splat red across RGB so a roughness or mask map reads the
                // same however the caller indexes it.
                let mut out = vec![255u8; (w * h * 4) as usize];
                for i in 0..(w * h) as usize {
                    let r = src.get(i).copied().unwrap_or(0);
                    out[i * 4] = r;
                    out[i * 4 + 1] = r;
                    out[i * 4 + 2] = r;
                }
                Texels::Ldr {
                    data: out,
                    srgb: false,
                }
            }
            TextureFormat::Rgba16Float => {
                let count = (w * h * 4) as usize;
                let mut out = vec![0.0f32; count];
                for (i, o) in out.iter_mut().enumerate() {
                    let lo = src.get(i * 2).copied().unwrap_or(0);
                    let hi = src.get(i * 2 + 1).copied().unwrap_or(0);
                    *o = f16_to_f32(u16::from_le_bytes([lo, hi]));
                }
                Texels::Hdr(out)
            }
            TextureFormat::Bc1RgbaUnormSrgb => Texels::Ldr {
                data: crate::textures::bc1::decode(src, w, h),
                srgb: true,
            },
            TextureFormat::Bc7RgbaUnormSrgb => return None,
        };
        let has_alpha = match &texels {
            Texels::Ldr { data, .. } => data.chunks_exact(4).any(|p| p[3] < 255),
            Texels::Hdr(data) => data.chunks_exact(4).any(|p| p[3] < 1.0),
        };
        Some(Self {
            width: w,
            height: h,
            texels,
            wrap_s: tex.wrap_s,
            wrap_t: tex.wrap_t,
            bilinear: tex.mag_filter == TextureFilter::Linear,
            flip_y: tex.flip_y,
            has_alpha,
            offset: tex.offset,
            repeat: tex.repeat,
            rotation: tex.rotation,
        })
    }

    /// Wrap linear f32 RGBA directly, with no conversion at all.
    ///
    /// This is the HDR environment path.
    /// [`CubeTexture::faces_f32`](crate::textures::CubeTexture::faces_f32)
    /// already holds exactly this, and routing it through the 8-bit `faces`
    /// instead would cost the dynamic range that makes an environment worth
    /// having — an HDRI's sun is thousands of times its sky, and the 8-bit copy
    /// is Reinhard-compressed and gamma-encoded specifically so it can be
    /// *displayed*, not integrated.
    pub fn from_linear_f32(
        width: u32,
        height: u32,
        data: Vec<f32>,
        bilinear: bool,
    ) -> Option<Self> {
        let want = (width as usize) * (height as usize) * 4;
        if width == 0 || height == 0 || data.len() < want {
            return None;
        }
        let has_alpha = data.chunks_exact(4).any(|p| p[3] < 1.0);
        Some(Self {
            width,
            height,
            texels: Texels::Hdr(data),
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            bilinear,
            flip_y: false,
            has_alpha,
            offset: Vector2::new(0.0, 0.0),
            repeat: Vector2::new(1.0, 1.0),
            rotation: 0.0,
        })
    }

    /// A 1×1 texture of one linear colour. Handy for tests and for standing in
    /// where a map is expected but absent.
    pub fn solid(rgba: [f32; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            texels: Texels::Hdr(rgba.to_vec()),
            wrap_s: TextureWrap::ClampToEdge,
            wrap_t: TextureWrap::ClampToEdge,
            bilinear: false,
            flip_y: false,
            has_alpha: rgba[3] < 1.0,
            offset: Vector2::new(0.0, 0.0),
            repeat: Vector2::new(1.0, 1.0),
            rotation: 0.0,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Linear RGBA of one texel, with no wrapping, filtering or UV transform.
    /// This is what the GPU backend copies into its atlas.
    pub fn texel_at(&self, x: u32, y: u32) -> [f32; 4] {
        self.texel(
            x.min(self.width.saturating_sub(1)),
            y.min(self.height.saturating_sub(1)),
        )
    }

    /// `(wrap_s, wrap_t)`.
    pub fn wrap_modes(&self) -> (TextureWrap, TextureWrap) {
        (self.wrap_s, self.wrap_t)
    }

    /// `(offset, repeat, rotation)` — the three.js UV transform.
    pub fn uv_transform(&self) -> (Vector2, Vector2, f32) {
        (self.offset, self.repeat, self.rotation)
    }

    /// Whether any texel is less than fully opaque.
    ///
    /// A base-colour map is *usually* opaque, and assuming it might not be
    /// costs every shadow ray in the scene the slow attenuating walk instead of
    /// a first-hit test. Scanning the decoded texture once settles it.
    pub fn has_transparency(&self) -> bool {
        self.has_alpha
    }

    /// Whether V is flipped on sample. The GPU atlas bakes this in when it
    /// copies, so the kernel never has to know.
    pub fn flips_y(&self) -> bool {
        self.flip_y
    }

    /// Sample at `uv`, returning linear RGBA.
    pub fn sample(&self, uv: Vector2) -> [f32; 4] {
        let (u, v) = self.transform_uv(uv);
        if self.bilinear {
            self.bilinear_at(u, v)
        } else {
            let x = self.wrap_x((u * self.width as f32).floor() as i64);
            let y = self.wrap_y((v * self.height as f32).floor() as i64);
            self.texel(x, y)
        }
    }

    /// Sample and drop alpha.
    pub fn sample_rgb(&self, uv: Vector2) -> Vector3 {
        let c = self.sample(uv);
        Vector3::new(c[0], c[1], c[2])
    }

    /// Apply the three.js transform chain: offset/repeat/rotation, then the
    /// V flip. `flip_y` exists because image decoders hand back rows top-first
    /// while GL-era UVs put V=0 at the bottom.
    fn transform_uv(&self, uv: Vector2) -> (f32, f32) {
        let (mut u, mut v) = (uv.x, uv.y);
        if self.rotation != 0.0 {
            let (s, c) = self.rotation.sin_cos();
            // Rotation is about (0.5, 0.5), matching three.js's default centre.
            let (du, dv) = (u - 0.5, v - 0.5);
            u = du * c - dv * s + 0.5;
            v = du * s + dv * c + 0.5;
        }
        u = u * self.repeat.x + self.offset.x;
        v = v * self.repeat.y + self.offset.y;
        if self.flip_y {
            v = 1.0 - v;
        }
        (u, v)
    }

    fn bilinear_at(&self, u: f32, v: f32) -> [f32; 4] {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let c00 = self.texel(self.wrap_x(x0), self.wrap_y(y0));
        let c10 = self.texel(self.wrap_x(x0 + 1), self.wrap_y(y0));
        let c01 = self.texel(self.wrap_x(x0), self.wrap_y(y0 + 1));
        let c11 = self.texel(self.wrap_x(x0 + 1), self.wrap_y(y0 + 1));
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let a = c00[c] * (1.0 - tx) + c10[c] * tx;
            let b = c01[c] * (1.0 - tx) + c11[c] * tx;
            out[c] = a * (1.0 - ty) + b * ty;
        }
        out
    }

    fn wrap_x(&self, x: i64) -> u32 {
        wrap_coord(x, self.width, self.wrap_s)
    }

    fn wrap_y(&self, y: i64) -> u32 {
        wrap_coord(y, self.height, self.wrap_t)
    }

    fn texel(&self, x: u32, y: u32) -> [f32; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        match &self.texels {
            Texels::Ldr { data, srgb } => {
                if i + 3 >= data.len() {
                    return [0.0, 0.0, 0.0, 1.0];
                }
                let lut = if *srgb { srgb_lut() } else { unit_lut() };
                [
                    lut[data[i] as usize],
                    lut[data[i + 1] as usize],
                    lut[data[i + 2] as usize],
                    // Alpha is never sRGB-encoded, in any format.
                    data[i + 3] as f32 * (1.0 / 255.0),
                ]
            }
            Texels::Hdr(data) => {
                if i + 3 >= data.len() {
                    return [0.0, 0.0, 0.0, 1.0];
                }
                [data[i], data[i + 1], data[i + 2], data[i + 3]]
            }
        }
    }
}

fn wrap_coord(c: i64, size: u32, mode: TextureWrap) -> u32 {
    let n = size.max(1) as i64;
    let w = match mode {
        TextureWrap::ClampToEdge => c.clamp(0, n - 1),
        TextureWrap::Repeat => c.rem_euclid(n),
        TextureWrap::MirroredRepeat => {
            let period = 2 * n;
            let m = c.rem_euclid(period);
            if m < n {
                m
            } else {
                period - 1 - m
            }
        }
    };
    w as u32
}

/// Pad or truncate to the length the dimensions imply, so a malformed texture
/// samples as black rather than panicking mid-render.
fn fit(src: &[u8], len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let n = src.len().min(len);
    out[..n].copy_from_slice(&src[..n]);
    out
}

/// sRGB → linear for all 256 byte values, built once.
fn srgb_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let s = i as f32 / 255.0;
            *v = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

/// The identity byte → unit-float table, so the linear path costs the same
/// lookup as the sRGB one instead of branching per channel.
fn unit_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        t
    })
}

/// IEEE-754 binary16 → binary32. Handles subnormals and infinities; NaN
/// payloads are not preserved, which nothing here depends on.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = match exp {
        0 => {
            if mant == 0 {
                sign << 31
            } else {
                // Subnormal: renormalise by shifting the mantissa up until the
                // implicit bit appears, paying for it in the exponent.
                let mut e = 0i32;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let exp32 = (127 - 15 + 1 + e) as u32;
                (sign << 31) | (exp32 << 23) | (m << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

/// Decodes each distinct [`Texture`] once, keyed by `Arc` identity — the same
/// rule the GPU renderer's upload cache uses, so a map shared by twenty
/// materials is expanded once.
#[derive(Debug, Default)]
pub struct TextureCache {
    by_ptr: HashMap<usize, Option<Arc<CpuTexture>>>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, tex: &Arc<Texture>) -> Option<Arc<CpuTexture>> {
        let key = Arc::as_ptr(tex) as usize;
        self.by_ptr
            .entry(key)
            .or_insert_with(|| CpuTexture::from_texture(tex).map(Arc::new))
            .clone()
    }

    /// Number of distinct textures decoded (including the ones that failed).
    pub fn len(&self) -> usize {
        self.by_ptr.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_ptr.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> Texture {
        // 2x2: black, white / white, black — in linear 8-bit.
        let data = vec![
            0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
        ];
        let mut t = Texture::new(2, 2, TextureFormat::Rgba8Unorm, data);
        t.mag_filter = TextureFilter::Nearest;
        t.flip_y = false;
        t
    }

    #[test]
    fn nearest_sampling_reads_the_expected_texel() {
        let tex = CpuTexture::from_texture(&checker()).unwrap();
        assert_eq!(tex.sample(Vector2::new(0.25, 0.25))[0], 0.0);
        assert_eq!(tex.sample(Vector2::new(0.75, 0.25))[0], 1.0);
        assert_eq!(tex.sample(Vector2::new(0.25, 0.75))[0], 1.0);
        assert_eq!(tex.sample(Vector2::new(0.75, 0.75))[0], 0.0);
    }

    #[test]
    fn bilinear_at_the_centre_averages_the_four() {
        let mut src = checker();
        src.mag_filter = TextureFilter::Linear;
        let tex = CpuTexture::from_texture(&src).unwrap();
        let c = tex.sample(Vector2::new(0.5, 0.5));
        assert!((c[0] - 0.5).abs() < 1e-3, "got {c:?}");
    }

    #[test]
    fn srgb_textures_decode_to_linear() {
        let mut src = Texture::new(
            1,
            1,
            TextureFormat::Rgba8UnormSrgb,
            vec![188, 188, 188, 255],
        );
        src.flip_y = false;
        let tex = CpuTexture::from_texture(&src).unwrap();
        let c = tex.sample(Vector2::new(0.5, 0.5));
        // 188/255 = 0.737 sRGB → ~0.5 linear.
        assert!((c[0] - 0.5).abs() < 0.02, "got {c:?}");
    }

    #[test]
    fn wrap_modes_behave() {
        assert_eq!(wrap_coord(-1, 4, TextureWrap::ClampToEdge), 0);
        assert_eq!(wrap_coord(9, 4, TextureWrap::ClampToEdge), 3);
        assert_eq!(wrap_coord(-1, 4, TextureWrap::Repeat), 3);
        assert_eq!(wrap_coord(5, 4, TextureWrap::Repeat), 1);
        assert_eq!(wrap_coord(4, 4, TextureWrap::MirroredRepeat), 3);
        assert_eq!(wrap_coord(-1, 4, TextureWrap::MirroredRepeat), 0);
    }

    #[test]
    fn half_floats_round_trip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 6.1e-5, 1e-7] {
            let bits = crate::renderer::gpu_texture::f32_to_f16_bits(v);
            let back = f16_to_f32(bits);
            // Relative for normals; one subnormal step (2^-24) as the floor,
            // because below ~6e-5 that step *is* the precision of the format.
            let tol = (v.abs() * 1e-3).max(6e-8);
            assert!((back - v).abs() <= tol, "{v} -> {back}");
        }
        // Subnormals decode to the right magnitude, not half or double it.
        assert!((f16_to_f32(0x0200) - 2f32.powi(-15)).abs() < 1e-12);
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-12);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn hdr_textures_keep_values_above_one() {
        let data = crate::textures::pack_rgba16f(&[4.0, 8.0, 16.0, 1.0]);
        let mut src = Texture::new(1, 1, TextureFormat::Rgba16Float, data);
        src.flip_y = false;
        let tex = CpuTexture::from_texture(&src).unwrap();
        let c = tex.sample(Vector2::new(0.5, 0.5));
        assert!((c[0] - 4.0).abs() < 0.01, "got {c:?}");
        assert!((c[2] - 16.0).abs() < 0.05, "got {c:?}");
    }

    #[test]
    fn transparency_is_detected_once_at_decode() {
        let opaque = CpuTexture::from_texture(&checker()).unwrap();
        assert!(!opaque.has_transparency());

        let mut src = checker();
        // Punch one texel through.
        let mut data = src.data.as_ref().clone();
        data[3] = 0;
        src.data = Arc::new(data);
        let cutout = CpuTexture::from_texture(&src).unwrap();
        assert!(cutout.has_transparency());

        assert!(CpuTexture::solid([1.0, 1.0, 1.0, 0.5]).has_transparency());
        assert!(!CpuTexture::solid([1.0, 1.0, 1.0, 1.0]).has_transparency());
    }

    #[test]
    fn cache_decodes_each_texture_once() {
        let a = Arc::new(checker());
        let b = Arc::clone(&a);
        let mut cache = TextureCache::new();
        let ta = cache.get(&a).unwrap();
        let tb = cache.get(&b).unwrap();
        assert!(Arc::ptr_eq(&ta, &tb));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn render_target_handles_have_no_cpu_pixels() {
        let handle = Texture::from_render_target(3, 64, 64, TextureFormat::Rgba8UnormSrgb);
        assert!(CpuTexture::from_texture(&handle).is_none());
    }
}
