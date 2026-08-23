use crate::textures::{
    CubeTexture, CubeUvAtlas, Texture, TextureFilter, TextureFormat, TextureWrap,
};
use std::sync::Arc;

/// WebGPU `write_texture` requires `bytes_per_row` to be a multiple of 256 when height > 1.
const TEXTURE_ROW_ALIGN: u32 = 256;

fn pad_rows_for_upload(pixels: &[u8], width: u32, height: u32, bpp: u32) -> (Vec<u8>, u32) {
    let unpadded = width * bpp;
    let padded = unpadded.div_ceil(TEXTURE_ROW_ALIGN) * TEXTURE_ROW_ALIGN;
    if padded == unpadded {
        return (pixels.to_vec(), padded);
    }
    let mut out = vec![0u8; (padded * height) as usize];
    for y in 0..height as usize {
        let src = y * unpadded as usize;
        let dst = y * padded as usize;
        out[dst..dst + unpadded as usize].copy_from_slice(&pixels[src..src + unpadded as usize]);
    }
    (out, padded)
}

/// IEEE-754 f32 → f16, rounding to nearest with ties to even.
///
/// The crate used to carry two of these — this one and a second inside the EXR
/// loader — which is exactly as safe as it sounds: this one masked the exponent
/// to fourteen bits where an f16 has fifteen, so every value from 2.0 up lost
/// its top exponent bit and came back a subnormal near zero, while the EXR copy
/// was correct the whole time. That reached `pack_rgba16f` and the PMREM atlas,
/// so every HDR environment map brighter than 2.0 was flattened before the GPU
/// saw it. The loader's version is now the only one, and both call it.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN — keep a NaN's payload non-zero so it stays a NaN.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    // Rebias 127 → 15.
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → zero
        }
        // Subnormal: shift the implicit 1 back in, then round.
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let round = ((m >> (shift - 1)) & 1) as u16;
        return sign | (half + round);
    }
    let half = ((e as u32) << 10) as u16 | (mant >> 13) as u16;
    let sticky = mant & 0x1fff;
    let round = u16::from(sticky > 0x1000 || (sticky == 0x1000 && (half & 1) == 1));
    sign | (half + round)
}

pub fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        if m == 0 {
            return f32::from_bits(s);
        }
        let mut e = 1u32;
        let mut m = m;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x3ff;
        return f32::from_bits(s | ((127 - 15 - e) << 23) | (m << 13));
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 127 - 15) << 23) | (m << 13))
}

fn pad_rows_rgba16f(f32_pixels: &[f32], width: u32, height: u32) -> (Vec<u8>, u32) {
    let bpp = 8u32;
    let unpadded = width * bpp;
    let padded = unpadded.div_ceil(TEXTURE_ROW_ALIGN) * TEXTURE_ROW_ALIGN;
    let mut out = vec![0u8; (padded * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let fi = ((y * width + x) * 4) as usize;
            let dst = (y * padded + x * bpp) as usize;
            for c in 0..4usize {
                let h = f32_to_f16_bits(f32_pixels[fi + c]);
                out[dst + c * 2..dst + c * 2 + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
    }
    (out, padded)
}

/// Builds mip chains on the GPU instead of the CPU.
///
/// Downsampling is a texture read and a write, which is what a GPU is; doing it
/// on the CPU means the whole chain is computed serially and then uploaded,
/// level by level, on whatever thread called `render`. In the browser that lands
/// on the main thread inside wasm on the first frame, and a planet's worth of
/// maps is tens of millions of texels of it.
///
/// The filter is one bilinear tap at the centre of each 2x2 block, which *is*
/// the box average, and it costs nothing. sRGB comes out right for free: the
/// sampler decodes on read and the render target encodes on write, so the
/// average happens in linear light without anyone converting anything.
pub struct MipGenerator {
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipelines: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers mipgen"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Vs { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

// One oversized triangle rather than two triangles: no seam down the diagonal
// and one fewer vertex to think about.
@vertex
fn vs(@builtin(vertex_index) i : u32) -> Vs {
    var out : Vs;
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    out.uv = uv;
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Clip space runs +y up and texture space +v down.
    out.pos.y = -out.pos.y;
    return out;
}

@group(0) @binding(0) var src  : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

@fragment
fn fs(in : Vs) -> @location(0) vec4<f32> {
    // The destination texel centre lands exactly between four source texels,
    // so a single bilinear tap returns their average.
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
"#
                .into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers mipgen layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers mipgen sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            shader,
            layout,
            sampler,
            pipelines: std::collections::HashMap::new(),
        }
    }

    /// Whether a format can be rendered into, and so filtered this way.
    fn renderable(format: wgpu::TextureFormat) -> bool {
        matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Rgba16Float
                | wgpu::TextureFormat::R8Unorm
        )
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let layout = &self.layout;
        let shader = &self.shader;
        self.pipelines.entry(format).or_insert_with(|| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers mipgen pipeline layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers mipgen"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs"),
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });
    }

    /// Fill levels `1..levels` of `texture` from level 0.
    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        levels: u32,
    ) {
        if levels < 2 {
            return;
        }
        // Built first, then everything borrowed immutably together — the
        // `&mut self` for pipeline creation must not overlap the reads below.
        self.ensure_pipeline(device, format);
        let pipeline = &self.pipelines[&format];
        let layout = &self.layout;
        let sampler = &self.sampler;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers mipgen"),
        });
        for level in 1..levels {
            let src = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let dst = texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers mipgen bind"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers mipgen pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// GPU-side handle for a 2D texture.
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// GPU-side handle for a cubemap (six faces) and optional CubeUV 2D atlas.
pub struct GpuCubeTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub cube_uv_texture: Option<wgpu::Texture>,
    pub cube_uv_view: Option<wgpu::TextureView>,
}

impl GpuCubeTexture {
    pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, src: &CubeTexture) -> Self {
        if let Some(atlas) = &src.cube_uv_atlas {
            return Self::upload_cube_uv(device, queue, src, atlas);
        }
        if let (Some(mips), Some(sizes)) = (&src.pmrem_mips, &src.pmrem_sizes) {
            return Self::upload_pmrem(device, queue, src.format, mips, sizes);
        }
        Self::upload_faces(device, queue, src.format, src.size, 1, &src.faces, 0)
    }

    fn upload_cube_uv(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &CubeTexture,
        atlas: &CubeUvAtlas,
    ) -> Self {
        let wgpu_fmt = wgpu_format(src.format);
        let cube = Self::upload_faces(device, queue, src.format, src.size.max(1), 1, &src.faces, 0);
        let (uv_fmt, upload_bytes, bytes_per_row) = if let Some(f32_px) = &atlas.pixels_f32 {
            let (bytes, bpr) = pad_rows_rgba16f(f32_px, atlas.width, atlas.height);
            (wgpu::TextureFormat::Rgba16Float, bytes, bpr)
        } else {
            let bpp = 4u32;
            let (bytes, bpr) =
                pad_rows_for_upload(atlas.pixels.as_ref(), atlas.width, atlas.height, bpp);
            (wgpu_fmt, bytes, bpr)
        };
        let uv_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers cube uv atlas"),
            size: wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: uv_fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(atlas.height),
            },
            wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
        );
        let uv_view = uv_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture: cube.texture,
            view: cube.view,
            cube_uv_texture: Some(uv_tex),
            cube_uv_view: Some(uv_view),
        }
    }

    fn upload_pmrem(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
        mips: &[[Arc<Vec<u8>>; 6]],
        sizes: &[u32],
    ) -> Self {
        let base_size = sizes[0].max(1);
        let mip_count = sizes.len() as u32;
        Self::upload_mip_faces(device, queue, format, base_size, mip_count, mips, sizes)
    }

    fn upload_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
        size: u32,
        mip_count: u32,
        faces: &[Arc<Vec<u8>>; 6],
        _base_level: u32,
    ) -> Self {
        let wgpu_fmt = wgpu_format(format);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers cube texture"),
            size: wgpu::Extent3d {
                width: size.max(1),
                height: size.max(1),
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Cube faces are built in memory by the environment and PMREM paths,
        // never loaded from a compressed file, so there is no block layout to
        // handle here -- but say so rather than let a wrong stride through.
        assert!(
            !format.is_block_compressed(),
            "cube textures are not block-compressed ({format:?})"
        );
        let bytes_per_pixel = format.bytes_per_block();
        let bytes_per_row = size * bytes_per_pixel as u32;
        for (layer, face) in faces.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                face,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        Self {
            texture,
            view,
            cube_uv_texture: None,
            cube_uv_view: None,
        }
    }

    fn upload_mip_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
        base_size: u32,
        mip_count: u32,
        mips: &[[Arc<Vec<u8>>; 6]],
        sizes: &[u32],
    ) -> Self {
        let wgpu_fmt = wgpu_format(format);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers pmrem cube texture"),
            size: wgpu::Extent3d {
                width: base_size,
                height: base_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Cube faces are built in memory by the environment and PMREM paths,
        // never loaded from a compressed file, so there is no block layout to
        // handle here -- but say so rather than let a wrong stride through.
        assert!(
            !format.is_block_compressed(),
            "cube textures are not block-compressed ({format:?})"
        );
        let bytes_per_pixel = format.bytes_per_block();
        for (level, (faces, size)) in mips.iter().zip(sizes.iter()).enumerate() {
            let size = *size;
            let bytes_per_row = size * bytes_per_pixel as u32;
            for (layer, face) in faces.iter().enumerate() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level as u32,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    face,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers pmrem cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        Self {
            texture,
            view,
            cube_uv_texture: None,
            cube_uv_view: None,
        }
    }
}

pub fn cube_cache_key(t: &Arc<CubeTexture>) -> *const CubeTexture {
    Arc::as_ptr(t)
}

/// Apply `Texture::flip_y` to the pixel data, borrowing when there is nothing
/// to do.
///
/// Image files store the top row first, but UV space puts `v = 0` at the
/// bottom, so a texture uploaded verbatim comes out mirrored vertically.
/// three.js resolves this by flipping on upload whenever `Texture.flipY` is set
/// — which is the default for everything except `DataTexture` and render
/// targets, both of which are authored bottom-up already. This crate declared
/// the same flag and then ignored it, which turned every equirectangular map
/// upside down: on a `SphereGeometry` the north pole is at `uv.y = 1`, so it
/// sampled the *last* row of the image.
fn oriented(src: &Texture) -> std::borrow::Cow<'_, Texture> {
    // Block-compressed data cannot be flipped by reversing rows: a row of BC
    // blocks holds four rows of pixels, and reversing them would shuffle the
    // image into quarters rather than turn it over. Flipping properly would
    // mean decoding and re-encoding, which is an offline job. So compressed
    // sources must arrive the right way up, and this refuses loudly instead of
    // returning a scrambled texture that looks almost plausible.
    if src.format.is_block_compressed() {
        assert!(
            !src.flip_y,
            "flip_y on a block-compressed texture ({:?}): rows of blocks cannot \
             be reversed -- encode it the right way up",
            src.format
        );
        return std::borrow::Cow::Borrowed(src);
    }
    let stride = src.width as usize * src.bytes_per_pixel();
    if !src.flip_y
        || src.external_rt_id.is_some()
        || stride == 0
        || src.data.len() != stride * src.height as usize
    {
        return std::borrow::Cow::Borrowed(src);
    }
    let mut data = Vec::with_capacity(src.data.len());
    for row in src.data.chunks_exact(stride).rev() {
        data.extend_from_slice(row);
    }
    let mut flipped = src.clone();
    flipped.data = std::sync::Arc::new(data);
    std::borrow::Cow::Owned(flipped)
}

impl GpuTexture {
    /// Upload, filling the mip chain on the GPU where the format allows it.
    ///
    /// Falls back to [`upload`](Self::upload) — which computes the chain on the
    /// CPU and uploads every level — for formats that cannot be rendered into.
    /// The two agree to within a level of 8-bit output; the GPU path averages
    /// through the sampler and the render target's own sRGB conversion, the CPU
    /// path through a table.
    pub fn upload_with_mipgen(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &Texture,
        mipgen: &mut MipGenerator,
    ) -> Self {
        let oriented = &*oriented(src);
        let format = wgpu_format(oriented.format);
        let (width, height) = (oriented.width.max(1), oriented.height.max(1));
        // A compressed texture cannot be a render attachment, so the GPU
        // mip generator has nothing to write into: it goes the plain route,
        // which uploads the levels it was given.
        if oriented.format.is_block_compressed() {
            return Self::upload(device, queue, src);
        }
        let bpp = oriented.bytes_per_pixel() as u32;
        let expected = (width as usize) * (height as usize) * bpp as usize;
        if !MipGenerator::renderable(format)
            || oriented.external_rt_id.is_some()
            || oriented.data.len() < expected
            || (width == 1 && height == 1)
        {
            return Self::upload(device, queue, src);
        }
        let levels = 1 + (width.max(height) as f32).log2().floor() as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &oriented.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bpp),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        mipgen.generate(device, queue, &texture, format, levels);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, src: &Texture) -> Self {
        // Orient before anything else, so the mip chain is built from the
        // pixels that actually get uploaded.
        let src = &*oriented(src);
        let format = wgpu_format(src.format);
        let (width, height) = (src.width.max(1), src.height.max(1));
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // A block-compressed texture's BASE size must be a whole number of
        // blocks. wgpu enforces it at create_texture — so this has to come
        // BEFORE that call, or its own validation error wins and names nothing:
        // "Width 2250 is not a multiple of ..." from inside a device call, with
        // no clue which of ninety textures it was. Mip LEVELS may be ragged; the
        // driver stores those padded.
        if src.format.is_block_compressed() {
            let (bw, bh) = src.format.block_dim();
            assert!(
                width % bw == 0 && height % bh == 0,
                "{:?} texture is {width}x{height}: a block-compressed base size \
                 must be a multiple of {bw}x{bh} (mip levels may be ragged, the \
                 base may not)",
                src.format
            );
        }
        // Every sampler in the renderer asks for trilinear filtering, so a
        // texture without a mip chain shimmers as soon as it is minified — and
        // the bigger the texture, the worse it gets. Build the chain up front.
        let mips = build_mip_chain(src);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers texture"),
            size,
            mip_level_count: 1 + mips.len() as u32 + src.mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (bw, bh) = src.format.block_dim();
        let write = |level: u32, w: u32, h: u32, data: &[u8]| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // In BLOCKS for compressed formats, which is what makes one
                    // `write` work for both: `bytes_per_row` counts a row of
                    // blocks and `rows_per_image` counts rows of them.
                    bytes_per_row: Some(src.format.bytes_per_row(w)),
                    rows_per_image: Some(src.format.rows_per_image(h)),
                },
                // PHYSICAL extent, not the logical one.
                //
                // A compressed mip level of 2x2 still occupies a whole 4x4
                // block, and wgpu validates the copy against that physical size
                // while requiring the width to be a whole number of blocks. Pass
                // the logical 2 and it rejects the copy outright -- which is why
                // the chain used to stop at the last block-aligned level and a
                // 14400-wide tile got four mips instead of twelve. Rounding up
                // here is what the format already does with the bytes.
                wgpu::Extent3d {
                    width: w.div_ceil(bw) * bw,
                    height: h.div_ceil(bh) * bh,
                    depth_or_array_layers: 1,
                },
            );
        };
        write(0, width, height, &src.data);
        for (level, (w, h, data)) in mips.iter().enumerate() {
            write(level as u32 + 1, *w, *h, data);
        }
        // Supplied levels, for formats whose chain cannot be derived. Sizes are
        // recomputed here rather than trusted, and a level whose length does not
        // match what the format says it should be is skipped with a complaint --
        // uploading it anyway reads past the end of somebody's buffer.
        let mut lw = width;
        let mut lh = height;
        for (i, level) in src.mips.iter().enumerate() {
            lw = (lw / 2).max(1);
            lh = (lh / 2).max(1);
            let want = src.format.data_len(lw, lh);
            if level.len() != want {
                eprintln!(
                    "threers: texture mip {} is {} bytes, expected {want} for {lw}x{lh} {:?} \
                     -- skipping the rest of the chain",
                    i + 1,
                    level.len(),
                    src.format
                );
                break;
            }
            write(i as u32 + 1, lw, lh, level);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}

/// Box-filtered mip levels below level 0, smallest last.
///
/// Empty when the texture is already 1×1, is render-target backed, carries the
/// wrong amount of data, or is a format this does not downsample
/// (`Rgba16Float` — HDR sources here are environment maps, which get their
/// chain from the PMREM path instead).
pub fn mip_chain_for(src: &Texture) -> usize {
    build_mip_chain(src).len()
}

fn build_mip_chain(src: &Texture) -> Vec<(u32, u32, Vec<u8>)> {
    // Compressed levels cannot be derived here -- box filtering BC blocks is
    // meaningless and the GPU cannot render into them either. They come from
    // `Texture::mips`, encoded offline; see the upload below.
    if src.format.is_block_compressed() {
        return Vec::new();
    }
    let bpp = src.bytes_per_pixel();
    let (w, h) = (src.width.max(1), src.height.max(1));
    if src.external_rt_id.is_some()
        || src.data.len() < (w as usize) * (h as usize) * bpp
        || (w == 1 && h == 1)
    {
        return Vec::new();
    }
    if matches!(src.format, TextureFormat::Rgba16Float) {
        return half_mip_chain(src);
    }
    build_mip_chain_8(src, w, h, bpp)
}

/// Half-float mip levels, averaged in the linear light the format already
/// stores.
///
/// Skipping these used to be defensible on the grounds that HDR sources are
/// environment maps, which get their chain from the PMREM path. A starfield is
/// not: it is sampled as an ordinary map on a sky sphere, and without a chain
/// the sampler has only level 0 to work with, so every star smaller than a
/// pixel is point-sampled — in frame at one camera angle, gone at the next.
/// From a vacuum that reads as twinkling.
fn half_mip_chain(src: &Texture) -> Vec<(u32, u32, Vec<u8>)> {
    let (mut w, mut h) = (src.width.max(1), src.height.max(1));
    let mut level: Vec<f32> = src
        .data
        .chunks_exact(2)
        .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let mut out = Vec::new();
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0f32; (nw as usize) * (nh as usize) * 4];
        for y in 0..nh as usize {
            let y0 = (y * 2).min(h as usize - 1);
            let y1 = (y * 2 + 1).min(h as usize - 1);
            for x in 0..nw as usize {
                let x0 = (x * 2).min(w as usize - 1);
                let x1 = (x * 2 + 1).min(w as usize - 1);
                let taps = [
                    (y0 * w as usize + x0) * 4,
                    (y0 * w as usize + x1) * 4,
                    (y1 * w as usize + x0) * 4,
                    (y1 * w as usize + x1) * 4,
                ];
                let dst = (y * nw as usize + x) * 4;
                for c in 0..4 {
                    next[dst + c] = taps.iter().map(|t| level[t + c]).sum::<f32>() * 0.25;
                }
            }
        }
        let bytes = next
            .iter()
            .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
            .collect();
        out.push((nw, nh, bytes));
        level = next;
        w = nw;
        h = nh;
    }
    out
}

fn build_mip_chain_8(src: &Texture, w: u32, h: u32, bpp: usize) -> Vec<(u32, u32, Vec<u8>)> {
    let (mut w, mut h) = (w, h);
    // sRGB values are not linear, so averaging the bytes directly darkens every
    // edge. Only the colour channels are encoded; alpha is already linear.
    let srgb = matches!(src.format, TextureFormat::Rgba8UnormSrgb);
    let color_channels = if bpp >= 3 { 3 } else { bpp };

    let decode = srgb_decode_table();
    let mut level: Vec<u8> = src.data.as_ref().clone();
    let mut out = Vec::new();
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0u8; (nw as usize) * (nh as usize) * bpp];
        for y in 0..nh as usize {
            // Odd sizes leave a final row/column with no partner; clamp to it.
            let y0 = (y * 2).min(h as usize - 1);
            let y1 = (y * 2 + 1).min(h as usize - 1);
            for x in 0..nw as usize {
                let x0 = (x * 2).min(w as usize - 1);
                let x1 = (x * 2 + 1).min(w as usize - 1);
                let taps = [
                    (y0 * w as usize + x0) * bpp,
                    (y0 * w as usize + x1) * bpp,
                    (y1 * w as usize + x0) * bpp,
                    (y1 * w as usize + x1) * bpp,
                ];
                let dst = (y * nw as usize + x) * bpp;
                for c in 0..bpp {
                    let encoded = srgb && c < color_channels;
                    let sum: f32 = taps
                        .iter()
                        .map(|t| {
                            let b = level[t + c];
                            if encoded {
                                decode[b as usize]
                            } else {
                                b as f32 / 255.0
                            }
                        })
                        .sum();
                    let avg = sum * 0.25;
                    let avg = if encoded {
                        linear_to_srgb_fast(avg)
                    } else {
                        avg
                    };
                    next[dst + c] = (avg * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out.push((nw, nh, next.clone()));
        level = next;
        w = nw;
        h = nh;
    }
    out
}

/// `srgb_to_linear` for every byte value, built once.
///
/// The mip builder's input is always a `u8`, so there are 256 possible answers
/// and no reason to call `powf` for any of them. A chain over a 1024x1024 sRGB
/// map does four decodes per output channel at every level — tens of millions
/// of `powf` calls for one texture — and the browser builds one of these per
/// upload, on the main thread, in wasm.
fn srgb_decode_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = srgb_to_linear(i as f32 / 255.0);
        }
        t
    })
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// `linear_to_srgb` over the curved part of the transfer function, sampled and
/// interpolated.
///
/// The decode table below cannot be mirrored directly — the encode's input is a
/// float, not a byte — but the curve is smooth above the linear toe, so a table
/// plus one lerp lands well inside a level of the 8-bit output it feeds. The
/// toe itself stays exact: it is a multiply.
///
/// This is one `powf` per output channel per mip level, and a chain over a
/// 4096x2048 map has eleven of those.
const SRGB_ENCODE_N: usize = 2048;
fn srgb_encode_table() -> &'static [f32; SRGB_ENCODE_N + 1] {
    static TABLE: std::sync::OnceLock<[f32; SRGB_ENCODE_N + 1]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; SRGB_ENCODE_N + 1];
        for (i, v) in t.iter_mut().enumerate() {
            let x = i as f32 / SRGB_ENCODE_N as f32;
            *v = 1.055 * x.powf(1.0 / 2.4) - 0.055;
        }
        t
    })
}

fn linear_to_srgb_fast(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    let x = v.min(1.0) * SRGB_ENCODE_N as f32;
    let i = x as usize;
    let f = x - i as f32;
    let t = srgb_encode_table();
    t[i] + (t[(i + 1).min(SRGB_ENCODE_N)] - t[i]) * f
}

/// The exact transfer function. Only the tests call it now — it is what
/// [`linear_to_srgb_fast`] is checked against, and keeping the definition
/// beside the approximation is the point.
#[cfg_attr(not(test), allow(dead_code))]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn wgpu_format(f: TextureFormat) -> wgpu::TextureFormat {
    match f {
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Bc1RgbaUnormSrgb => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        TextureFormat::Bc7RgbaUnormSrgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    }
}

pub fn wgpu_filter(f: TextureFilter) -> wgpu::FilterMode {
    match f {
        TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        TextureFilter::Linear => wgpu::FilterMode::Linear,
    }
}

pub fn wgpu_wrap(w: TextureWrap) -> wgpu::AddressMode {
    match w {
        TextureWrap::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        TextureWrap::Repeat => wgpu::AddressMode::Repeat,
        TextureWrap::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

/// What makes two `Texture`s the same *upload*.
///
/// Keyed on the pixel data rather than on the `Texture` object, so several
/// textures over one buffer — the same map read through different UV
/// transforms, say — cost one GPU texture between them instead of one each.
/// The Moon's eight patches used to slice twenty-four copies out of three maps
/// and upload all of them; sharing turns that into three.
///
/// Everything that changes the uploaded bytes is in the key. `wrap` and the
/// filters are not: they select a sampler at bind time and never touch the
/// texture, so two views of one buffer may wrap differently and still share it.
pub type TexCacheKey = (u64, u64, u32, u32, u32, bool);

pub fn tex_cache_key(t: &Arc<Texture>) -> TexCacheKey {
    let format = match t.format {
        TextureFormat::Rgba8Unorm => 0,
        TextureFormat::Rgba8UnormSrgb => 1,
        TextureFormat::R8Unorm => 2,
        TextureFormat::Rgba16Float => 3,
        TextureFormat::Bc1RgbaUnormSrgb => 4,
        TextureFormat::Bc7RgbaUnormSrgb => 5,
    };
    // `flip_y` is in the key because the upload honours it — two textures over
    // the same bytes with opposite flips are genuinely different images. The
    // size and format are in it because `Texture`'s fields are public and can
    // be changed after construction.
    //
    // `id` is the stable logical texture; `upload_seq` bumps when bytes are
    // replaced in place (video frames) so the renderer re-uploads without minting
    // a new id every frame.
    (t.id, t.upload_seq, t.width, t.height, format, t.flip_y)
}

#[cfg(test)]
mod mip_tests {
    use super::*;
    use crate::textures::{Texture, TextureFormat};

    #[test]
    fn the_fast_srgb_encode_matches_the_exact_one() {
        // The mip builder encodes once per output channel per level, and a
        // chain over a 4096x2048 map has eleven of them, so this replaced a
        // `powf` with a table and a lerp. It has to stay inside the 8-bit
        // output it feeds or mips drift away from their level 0.
        let mut worst = 0.0f32;
        for i in 0..=20000 {
            let v = i as f32 / 20000.0;
            worst = worst.max((linear_to_srgb_fast(v) - linear_to_srgb(v)).abs() * 255.0);
        }
        assert!(
            worst < 0.5,
            "fast sRGB encode is off by {worst} levels of 255"
        );
    }

    #[test]
    fn half_float_survives_the_round_trip_above_one() {
        // The exponent field was masked to 14 bits instead of 15, so the top
        // exponent bit fell off and everything from 2.0 up came back as a
        // subnormal — 5.0 arrived as 7.6e-5. HDR is the entire point of this
        // format, and this path feeds the PMREM atlas and `pack_rgba16f`, so
        // every environment map brighter than 2.0 was being flattened.
        for v in [
            0.0, 1e-4, 0.1, 0.5, 1.0, 1.9, 2.0, 2.1, 5.0, 20.6, 100.0, 550.0, 6000.0, 60000.0,
        ] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            let tol = (v * 0.002).max(1e-6);
            assert!(
                (back - v).abs() <= tol,
                "{v} round-tripped to {back}, off by more than {tol}"
            );
        }
        for v in [-1.0f32, -3.5, -900.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!(
                (back - v).abs() <= (v.abs() * 0.002).max(1e-6),
                "{v} round-tripped to {back}"
            );
        }
        // Past the format's range it should saturate, not wrap to something small.
        assert!(f16_bits_to_f32(f32_to_f16_bits(1e9)) > 60000.0);
    }

    #[test]
    fn half_float_maps_get_a_mip_chain() {
        // Skipping these left the sampler with only level 0, so a starfield
        // point-sampled its stars: in frame at one camera angle, gone at the
        // next, which from vacuum reads as twinkling.
        let texels = 8 * 4;
        let mut data = Vec::with_capacity(texels * 8);
        for _ in 0..texels {
            for c in [4.0f32, 2.0, 1.0, 1.0] {
                data.extend_from_slice(&f32_to_f16_bits(c).to_le_bytes());
            }
        }
        let src = Texture::new(8, 4, TextureFormat::Rgba16Float, data);
        let mips = build_mip_chain(&src);
        let sizes: Vec<(u32, u32)> = mips.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(sizes, vec![(4, 2), (2, 1), (1, 1)]);
        // Averaging a constant map has to give the constant back — and at 4.0,
        // which is where the exponent bug used to bite.
        let (_, _, last) = mips.last().unwrap();
        let red = f16_bits_to_f32(u16::from_le_bytes([last[0], last[1]]));
        assert!((red - 4.0).abs() < 0.01, "1x1 mip came out at {red}");
    }

    #[test]
    fn a_chain_runs_down_to_one_by_one() {
        let src = Texture::new(8, 4, TextureFormat::Rgba8Unorm, vec![128; 8 * 4 * 4]);
        let mips = build_mip_chain(&src);
        let sizes: Vec<(u32, u32)> = mips.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(sizes, vec![(4, 2), (2, 1), (1, 1)]);
        for (w, h, data) in &mips {
            assert_eq!(data.len(), (*w as usize) * (*h as usize) * 4);
        }
    }

    #[test]
    fn non_power_of_two_sizes_still_terminate() {
        let src = Texture::new(5, 3, TextureFormat::Rgba8Unorm, vec![7; 5 * 3 * 4]);
        let sizes: Vec<(u32, u32)> = build_mip_chain(&src)
            .iter()
            .map(|(w, h, _)| (*w, *h))
            .collect();
        assert_eq!(sizes, vec![(2, 1), (1, 1)]);
    }

    #[test]
    fn a_flat_image_downsamples_to_the_same_value() {
        for format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
            let src = Texture::new(4, 4, format, vec![200; 4 * 4 * 4]);
            for (_, _, data) in build_mip_chain(&src) {
                assert!(
                    data.iter().all(|&v| (v as i32 - 200).abs() <= 1),
                    "{format:?} drifted: {data:?}"
                );
            }
        }
    }

    #[test]
    fn srgb_averages_in_linear_light() {
        // Half black, half white. Averaging the *bytes* gives 128; averaging
        // the light they represent gives ~188, which is the correct answer and
        // the reason edges do not darken at distance.
        let mut data = vec![0u8; 8];
        data[4..8].copy_from_slice(&[255, 255, 255, 255]);
        let srgb = Texture::new(2, 1, TextureFormat::Rgba8UnormSrgb, data.clone());
        let (_, _, mip) = build_mip_chain(&srgb).remove(0);
        assert!((mip[0] as i32 - 188).abs() <= 2, "sRGB mip was {}", mip[0]);

        let linear = Texture::new(2, 1, TextureFormat::Rgba8Unorm, data);
        let (_, _, mip) = build_mip_chain(&linear).remove(0);
        assert!(
            (mip[0] as i32 - 128).abs() <= 1,
            "linear mip was {}",
            mip[0]
        );
    }

    #[test]
    fn sources_without_pixels_to_filter_are_skipped() {
        // Half-float used to be on this list, on the grounds that HDR sources
        // are environment maps and get their chain from PMREM. A starfield is
        // not — see `half_float_maps_get_a_mip_chain`.
        let rt = Texture::from_render_target(1, 64, 64, TextureFormat::Rgba8Unorm);
        assert!(build_mip_chain(&rt).is_empty());
        // Truncated data must not panic or read out of bounds.
        let short = Texture::new(16, 16, TextureFormat::Rgba8Unorm, vec![0; 8]);
        assert!(build_mip_chain(&short).is_empty());
    }

    #[test]
    fn single_channel_textures_chain_too() {
        let src = Texture::new(4, 4, TextureFormat::R8Unorm, vec![64; 16]);
        let mips = build_mip_chain(&src);
        assert_eq!(mips.len(), 2);
        assert!(mips[0].2.iter().all(|&v| v == 64));
    }
}
