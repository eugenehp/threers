//! Volumes on the CPU ([`Data3DTexture`]) and on the GPU ([`GpuVolume`]).

/// How a volume's texels are stored.
///
/// The integer formats are for labels, where a value is an identity rather
/// than an intensity and must come back exactly: a segmentation, an atlas, a
/// material map. `Rg32Uint` holds a 64-bit label as low and high words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolumeFormat {
    R8Unorm,
    Rgba8Unorm,
    R16Float,
    R32Float,
    R8Uint,
    R16Uint,
    R32Uint,
    /// A `u64` per texel, little-endian: `.r` is the low word, `.g` the high.
    Rg32Uint,
}

/// What a shader reads back from a volume: floats or unsigned integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolumeSample {
    /// `texture_3d<f32>`, read with `textureLoad` as `vec4<f32>`.
    Float,
    /// `texture_3d<u32>`, read with `textureLoad` as `vec4<u32>`.
    Uint,
}

impl VolumeFormat {
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            VolumeFormat::R8Unorm | VolumeFormat::R8Uint => 1,
            VolumeFormat::R16Float | VolumeFormat::R16Uint => 2,
            VolumeFormat::Rgba8Unorm | VolumeFormat::R32Float | VolumeFormat::R32Uint => 4,
            VolumeFormat::Rg32Uint => 8,
        }
    }

    pub fn sample(self) -> VolumeSample {
        match self {
            VolumeFormat::R8Unorm
            | VolumeFormat::Rgba8Unorm
            | VolumeFormat::R16Float
            | VolumeFormat::R32Float => VolumeSample::Float,
            VolumeFormat::R8Uint
            | VolumeFormat::R16Uint
            | VolumeFormat::R32Uint
            | VolumeFormat::Rg32Uint => VolumeSample::Uint,
        }
    }

    pub fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            VolumeFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            VolumeFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            VolumeFormat::R16Float => wgpu::TextureFormat::R16Float,
            VolumeFormat::R32Float => wgpu::TextureFormat::R32Float,
            VolumeFormat::R8Uint => wgpu::TextureFormat::R8Uint,
            VolumeFormat::R16Uint => wgpu::TextureFormat::R16Uint,
            VolumeFormat::R32Uint => wgpu::TextureFormat::R32Uint,
            VolumeFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        }
    }
}

/// three.js `Data3DTexture`: a `width × height × depth` block of texels,
/// x fastest, then y, then z (`x + y·width + z·width·height`).
///
/// This is the CPU side; [`GpuVolume::from_texture`] uploads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Data3DTexture {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: VolumeFormat,
}

impl Data3DTexture {
    /// # Panics
    ///
    /// If `data` is not exactly `width · height · depth` texels long.
    pub fn new(width: u32, height: u32, depth: u32, format: VolumeFormat, data: Vec<u8>) -> Self {
        let expected = width as usize * height as usize * depth as usize * format.bytes_per_texel() as usize;
        assert_eq!(
            data.len(),
            expected,
            "Data3DTexture {width}x{height}x{depth} {format:?} needs {expected} bytes"
        );
        Self {
            data,
            width,
            height,
            depth,
            format,
        }
    }

    /// An `R32Float` volume from `f32` values.
    pub fn from_f32(width: u32, height: u32, depth: u32, values: &[f32]) -> Self {
        Self::new(width, height, depth, VolumeFormat::R32Float, bytemuck::cast_slice(values).to_vec())
    }

    /// An `Rg32Uint` volume from `u64` labels.
    pub fn from_u64(width: u32, height: u32, depth: u32, labels: &[u64]) -> Self {
        let data = labels.iter().flat_map(|l| l.to_le_bytes()).collect();
        Self::new(width, height, depth, VolumeFormat::Rg32Uint, data)
    }

    pub fn size(&self) -> [u32; 3] {
        [self.width, self.height, self.depth]
    }
}

/// A volume in a 3D texture on the device.
///
/// Made whole from a [`Data3DTexture`], or empty and then filled a region at a
/// time with [`write_region`](Self::write_region) — which is how a volume too
/// large to load at once is streamed in as its chunks arrive.
pub struct GpuVolume {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: [u32; 3],
    pub format: VolumeFormat,
}

impl GpuVolume {
    /// An uninitialized volume (texels read as zero until written).
    pub fn new(device: &wgpu::Device, size: [u32; 3], format: VolumeFormat) -> Self {
        let [w, h, d] = size.map(|n| n.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers volume"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: d,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: format.wgpu(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size: [w, h, d],
            format,
        }
    }

    pub fn from_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &Data3DTexture) -> Self {
        let volume = Self::new(device, texture.size(), texture.format);
        volume.write_region(queue, [0, 0, 0], texture.size(), &texture.data);
        volume
    }

    /// Replaces the box `origin .. origin + size` with `bytes`, laid out like a
    /// [`Data3DTexture`] of that size and this volume's format.
    ///
    /// No row alignment is needed: this is `Queue::write_texture`, not a
    /// buffer copy.
    ///
    /// # Panics
    ///
    /// If the box is not inside the volume or `bytes` is the wrong length.
    pub fn write_region(&self, queue: &wgpu::Queue, origin: [u32; 3], size: [u32; 3], bytes: &[u8]) {
        for axis in 0..3 {
            assert!(
                origin[axis] + size[axis] <= self.size[axis],
                "region {origin:?}+{size:?} is outside the {:?} volume",
                self.size
            );
        }
        let bpt = self.format.bytes_per_texel();
        let [w, h, d] = size;
        assert_eq!(
            bytes.len(),
            (w * h * d * bpt) as usize,
            "region {size:?} of {:?} needs {} bytes",
            self.format,
            w * h * d * bpt
        );
        if w == 0 || h == 0 || d == 0 {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin[0],
                    y: origin[1],
                    z: origin[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpt),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: d,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u64_labels_are_little_endian_word_pairs() {
        let t = Data3DTexture::from_u64(1, 1, 1, &[0x0000_0002_0000_0001]);
        assert_eq!(t.data, [1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(t.format.sample(), VolumeSample::Uint);
    }

    #[test]
    #[should_panic(expected = "needs 24 bytes")]
    fn wrong_length_is_rejected() {
        Data3DTexture::new(2, 3, 1, VolumeFormat::R32Float, vec![0; 20]);
    }
}
