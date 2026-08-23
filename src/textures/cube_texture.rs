use super::{TextureFilter, TextureFormat, TextureWrap};
use std::sync::Arc;

/// CPU-side CubeUV PMREM atlas (three.js `CubeUVReflectionMapping` layout).
#[derive(Debug, Clone)]
pub struct CubeUvAtlas {
    pub width: u32,
    pub height: u32,
    pub cube_size: u32,
    pub lod_max: u32,
    pub texel_width: f32,
    pub texel_height: f32,
    pub pixels: Arc<Vec<u8>>,
    /// Linear f32 RGBA atlas before 8-bit quantize (PMREM blur output). GPU uploads this.
    pub pixels_f32: Option<Arc<Vec<f32>>>,
}

/// Cubemap with six faces in the order +X, -X, +Y, -Y, +Z, -Z.
/// Each face must have the same width and height.
#[derive(Debug, Clone)]
pub struct CubeTexture {
    pub size: u32,
    pub format: TextureFormat,
    pub mag_filter: TextureFilter,
    pub min_filter: TextureFilter,
    pub wrap: TextureWrap,
    /// 6 faces. Each face's length must equal `size * size * bytes_per_pixel`.
    pub faces: [Arc<Vec<u8>>; 6],
    /// Optional **linear f32** RGBA face data, in the same face order.
    ///
    /// This is the HDR input path. The PMREM blur chain already runs in f32
    /// internally and uploads its atlas as `Rgba16Float`, so when this is
    /// present values above 1.0 survive prefiltering instead of being clamped
    /// by the 8-bit `faces`. `faces` is still populated (tone-mapped down) so
    /// the non-PMREM cube-sampling path and any 8-bit consumer keep working.
    pub faces_f32: Option<[Arc<Vec<f32>>; 6]>,
    /// Optional PMREM mip chain: each entry is six faces at decreasing resolution.
    /// When present, GPU upload writes a full mip pyramid for roughness-based IBL.
    pub pmrem_mips: Option<Vec<[Arc<Vec<u8>>; 6]>>,
    pub pmrem_sizes: Option<Vec<u32>>,
    /// CubeUV PMREM atlas (three.js `CubeUVReflectionMapping` layout).
    pub cube_uv_atlas: Option<CubeUvAtlas>,
}

impl CubeTexture {
    pub fn new(size: u32, format: TextureFormat, faces: [Vec<u8>; 6]) -> Self {
        let [f0, f1, f2, f3, f4, f5] = faces;
        Self {
            size,
            format,
            mag_filter: TextureFilter::Linear,
            min_filter: TextureFilter::Linear,
            wrap: TextureWrap::ClampToEdge,
            faces: [
                Arc::new(f0),
                Arc::new(f1),
                Arc::new(f2),
                Arc::new(f3),
                Arc::new(f4),
                Arc::new(f5),
            ],
            faces_f32: None,
            pmrem_mips: None,
            pmrem_sizes: None,
            cube_uv_atlas: None,
        }
    }

    /// Build an **HDR** cubemap from linear f32 RGBA faces (`+X, -X, +Y, -Y,
    /// +Z, -Z`), each `size * size * 4` floats long.
    ///
    /// Values above 1.0 are preserved for prefiltering — this is what lets an
    /// environment hold a sun several thousand times brighter than its sky. A
    /// clamped 8-bit copy is generated alongside so every existing consumer
    /// (raw cube sampling, mip export) still works; pair this with a tone
    /// mapper or the highlights will simply clip on output instead.
    pub fn new_f32(size: u32, faces: [Vec<f32>; 6]) -> Self {
        let expect = (size as usize) * (size as usize) * 4;
        let clamped: [Vec<u8>; 6] = std::array::from_fn(|i| {
            let f = &faces[i];
            let mut out = Vec::with_capacity(expect);
            for px in f.chunks(4) {
                // Reinhard the 8-bit fallback rather than hard-clipping, so the
                // low-dynamic-range copy still shows structure inside the sun.
                for c in 0..3 {
                    let v = px.get(c).copied().unwrap_or(0.0).max(0.0);
                    let m = v / (1.0 + v);
                    out.push((m.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8);
                }
                out.push((px.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0) * 255.0) as u8);
            }
            out.resize(expect, 0);
            out
        });
        let mut cube = Self::new(size, TextureFormat::Rgba8UnormSrgb, clamped);
        cube.faces_f32 = Some(std::array::from_fn(|i| Arc::new(faces[i].clone())));
        cube
    }

    pub fn with_pmrem_mips(mut self, mips: Vec<[Vec<u8>; 6]>, sizes: Vec<u32>) -> Self {
        self.pmrem_mips = Some(
            mips.into_iter()
                .map(|f| {
                    let [a, b, c, d, e, g] = f;
                    [
                        Arc::new(a),
                        Arc::new(b),
                        Arc::new(c),
                        Arc::new(d),
                        Arc::new(e),
                        Arc::new(g),
                    ]
                })
                .collect(),
        );
        self.pmrem_sizes = Some(sizes);
        self
    }

    pub fn with_cube_uv_atlas(mut self, atlas: CubeUvAtlas) -> Self {
        self.cube_uv_atlas = Some(atlas);
        self
    }

    pub fn is_cube_uv(&self) -> bool {
        self.cube_uv_atlas.is_some()
    }

    pub fn mip_level_count(&self) -> u32 {
        self.pmrem_sizes
            .as_ref()
            .map(|s| s.len() as u32)
            .unwrap_or(1)
    }
}
