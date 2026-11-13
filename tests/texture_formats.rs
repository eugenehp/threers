//! Every texture format against the GPU that samples it.
//!
//! The block decoders in `textures::codec` exist for GPUs that cannot sample a
//! format — BC on phones, ETC2 and ASTC on desktops — so what they produce has
//! to be what a GPU that *can* would have shown. Here a GPU that can is the
//! reference: each format's blocks are uploaded natively, read back texel by
//! texel through `textureLoad`, and compared with the CPU decode of the same
//! blocks. The blocks are random but steered so every mode of every format
//! appears (and, for ASTC, so most blocks are valid at all, which random bits
//! almost never are).
//!
//! Then the same through a device with none of the compression features,
//! where the upload has to decode on the CPU: it must come out as the native
//! device sampled it.
//!
//! Each part skips with a note when the adapter lacks the family; on Apple
//! silicon every family is present.

use threers::textures::codec::{self, Support};
use threers::textures::{AstcBlock, Compression, Texture, TextureFormat};
use wgpu::util::DeviceExt;

const WANTED: wgpu::Features = wgpu::Features::TEXTURE_COMPRESSION_BC
    .union(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
    .union(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
    .union(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    .union(wgpu::Features::FLOAT32_FILTERABLE);

fn device(what: &str, features: wgpu::Features) -> Option<threers::gpu::GpuContext> {
    match pollster::block_on(threers::gpu::GpuContext::headless_with_optional_features(
        features,
    )) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping {what}: {e}");
            None
        }
    }
}

/// A small deterministic generator, so a failure reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Read every texel of `view` (level 0) back as RGBA f32.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    w: u32,
    h: u32,
) -> Vec<[f32; 4]> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("texel readback"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var t: texture_2d<f32>;
             @group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
             @compute @workgroup_size(8, 8)
             fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                 let size = textureDimensions(t);
                 if id.x < size.x && id.y < size.y {
                     out[id.y * size.x + id.x] = textureLoad(t, vec2<i32>(id.xy), 0);
                 }
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = (w * h * 16) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &readback, 0, size);
    queue.submit([enc.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let floats: Vec<f32> =
        bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
    floats.as_chunks::<4>().0.to_vec()
}

/// Upload `data` natively as `format` (level 0 only) and read it back.
fn gpu_decode(
    gpu: &threers::gpu::GpuContext,
    format: TextureFormat,
    w: u32,
    h: u32,
    data: &[u8],
) -> Vec<[f32; 4]> {
    let wf = threers::renderer::gpu_texture::wgpu_format(format);
    let texture = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wf,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        data,
    );
    read_back(
        &gpu.device,
        &gpu.queue,
        &texture.create_view(&Default::default()),
        w,
        h,
    )
}

fn srgb_encode(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// The CPU decode of `data` as RGBA f32 in the same terms the GPU's
/// `textureLoad` returns: unorm as `[0, 1]`, missing channels 0 and alpha 1,
/// and for sRGB formats still sRGB-encoded (the GPU side is re-encoded).
fn cpu_decode(format: TextureFormat, w: u32, h: u32, data: &[u8]) -> Vec<[f32; 4]> {
    let decoded = codec::decoded_format(format);
    let bytes = if format.is_block_compressed() {
        codec::decode(format, data, w, h).expect("decodes")
    } else {
        data.to_vec()
    };
    let v = decoded.texels_f32(&bytes).expect("texels");
    let n = decoded.channels();
    (0..(w * h) as usize)
        .map(|i| {
            let t = &v[i * n..i * n + n];
            match n {
                1 => [t[0], 0.0, 0.0, 1.0],
                2 => [t[0], t[1], 0.0, 1.0],
                _ => [t[0], t[1], t[2], t[3]],
            }
        })
        .collect()
}

/// Compare, in 8-bit steps for unorm formats and in half-float ulps for
/// BC6H; the result is the worst difference and how many texels exceed `tol`.
fn compare(
    format: TextureFormat,
    cpu: &[[f32; 4]],
    gpu: &[[f32; 4]],
    tol: f32,
) -> (f32, usize, Vec<String>) {
    let srgb = format.is_srgb();
    let hdr = matches!(
        format,
        TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat
    );
    let mut worst = 0f32;
    let mut bad = 0;
    let mut examples = Vec::new();
    for (i, (c, g)) in cpu.iter().zip(gpu).enumerate() {
        let mut d = 0f32;
        for k in 0..4 {
            let (cv, mut gv) = (c[k], g[k]);
            if srgb && k < 3 {
                gv = srgb_encode(gv);
            }
            d = d.max(if hdr {
                // Distance in representable halves.
                let ch = threers::math::float::f32_to_f16(cv) as i32;
                let gh = threers::math::float::f32_to_f16(gv) as i32;
                let key = |h: i32| if h & 0x8000 != 0 { -(h & 0x7fff) } else { h };
                (key(ch) - key(gh)).abs() as f32
            } else {
                (cv - gv).abs() * 255.0
            });
        }
        worst = worst.max(d);
        if d > tol {
            bad += 1;
            if examples.len() < 6 {
                examples.push(format!("texel {i}: cpu {c:?} gpu {g:?}"));
            }
        }
    }
    (worst, bad, examples)
}

/// Random blocks for `format`, `n` of them, steered to cover every mode.
fn blocks(format: TextureFormat, n: usize, rng: &mut Rng) -> Vec<u8> {
    let bytes = format.bytes_per_block();
    let mut out = Vec::with_capacity(n * bytes);
    for i in 0..n {
        let mut b = rng.bytes(bytes);
        match format {
            // BC7: the mode is the position of the first set bit; spread them
            // evenly, and include the reserved all-zero byte.
            TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => {
                let mode = i % 9;
                b[0] = if mode == 8 {
                    0
                } else {
                    (b[0] & !((2u8 << mode) - 1)) | (1 << mode)
                };
            }
            // BC6H: every mode code, the reserved ones included.
            TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => {
                const CODES: [u8; 18] = [
                    0, 1, 0x02, 0x06, 0x0a, 0x0e, 0x12, 0x16, 0x1a, 0x1e, 0x03, 0x07, 0x0b, 0x0f,
                    0x13, 0x17, 0x1b, 0x1f,
                ];
                let c = CODES[i % CODES.len()];
                b[0] = if c < 2 {
                    (b[0] & !3) | c
                } else {
                    (b[0] & !0x1f) | c
                };
            }
            TextureFormat::Astc { block, .. } => b = astc_block(block, i, rng),
            _ => {}
        }
        out.extend_from_slice(&b);
    }
    out
}

/// An ASTC block: one in eight left random (almost always malformed, so the
/// error colour is compared too), one in eight a void extent, the rest valid
/// blocks with a random grid, partition count and LDR endpoint modes.
fn astc_block(block: AstcBlock, i: usize, rng: &mut Rng) -> Vec<u8> {
    let (bw, bh) = block.dims();
    loop {
        let mut b = rng.bytes(16);
        match i % 8 {
            0 => return b,
            1 => {
                // Void extent, LDR: block mode 0x1fc, bits 9 = 0, 10-11 = 1,
                // extents all ones.
                let v = u128::from_le_bytes(b.clone().try_into().unwrap());
                let colour = v & !((1u128 << 64) - 1);
                let header = 0x1fcu128 | (3 << 10) | (((1u128 << 52) - 1) << 12);
                return (colour | header).to_le_bytes().to_vec();
            }
            _ => {}
        }
        // A random block mode and partition count, then an LDR endpoint mode
        // (or several), then keep it if the decoder finds it well formed —
        // which the comparison then holds to the hardware.
        let mode = (rng.next() & 0x7ff) as u128;
        let partitions = (rng.next() % 4) as u128;
        let ldr = [0u128, 1, 4, 5, 6, 8, 9, 10, 12, 13];
        let mut v = u128::from_le_bytes(b.clone().try_into().unwrap());
        v = (v & !0x1fff) | mode | (partitions << 11);
        if partitions == 0 {
            let cem = ldr[(rng.next() % ldr.len() as u64) as usize];
            v = (v & !(0xf << 13)) | (cem << 13);
        } else if rng.next().is_multiple_of(2) {
            let cem = ldr[(rng.next() % ldr.len() as u64) as usize];
            v = (v & !(0x3f << 23)) | (cem << 25);
        }
        b = v.to_le_bytes().to_vec();
        let mut px = vec![0u8; (bw * bh * 4) as usize];
        let t = codec::decode(TextureFormat::Astc { block, srgb: false }, &b, bw, bh).unwrap();
        px.copy_from_slice(&t);
        let error = px
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [255, 0, 255, 255]);
        if !error {
            return b;
        }
    }
}

/// The tolerance each format is held to, in 8-bit steps. 0 where the decode
/// is exact, which is everything but these:
/// - BC4 and BC5 (and BC3's alpha) come back from the GPU unrounded, and the
///   8-bit decode must round; a value exactly between two steps is half a step
///   from either.
/// - Linear ASTC is a 16-bit value the GPU hands over as a half float, which
///   adds a sliver to the same half step (0.553 measured).
/// - sRGB formats lose up to half a step to the comparison's own round trip
///   through linear (0.39 measured).
fn tolerance(format: TextureFormat) -> f32 {
    use TextureFormat::*;
    let t = match format {
        Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc4RUnorm | Bc5RgUnorm => 0.51,
        Astc { srgb: false, .. } => 0.6,
        _ => 0.0,
    };
    if format.is_srgb() {
        t + 0.5
    } else {
        t
    }
}

#[test]
fn every_block_format_decodes_as_the_gpu_does() {
    let Some(gpu) = device("block formats", WANTED) else {
        return;
    };
    let features = gpu.device.features();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut report = Vec::new();
    let mut failed = Vec::new();
    let mut skipped = Vec::new();
    for format in TextureFormat::all()
        .into_iter()
        .filter(|f| f.is_block_compressed())
    {
        let needed = match format.compression().unwrap() {
            Compression::Bc => wgpu::Features::TEXTURE_COMPRESSION_BC,
            Compression::Etc2 => wgpu::Features::TEXTURE_COMPRESSION_ETC2,
            Compression::Astc => wgpu::Features::TEXTURE_COMPRESSION_ASTC,
        };
        if !features.contains(needed) {
            skipped.push(format!("{format:?}"));
            continue;
        }
        let (bw, bh) = format.block_dim();
        let (nx, ny) = (16u32, 16u32);
        let (w, h) = (nx * bw, ny * bh);
        let data = blocks(format, (nx * ny) as usize, &mut rng);
        let cpu = cpu_decode(format, w, h, &data);
        let hw = gpu_decode(&gpu, format, w, h, &data);
        let (worst, bad, examples) = compare(format, &cpu, &hw, tolerance(format));
        report.push(format!(
            "{format:?}: worst {worst}, {bad} of {} over {}",
            cpu.len(),
            tolerance(format)
        ));
        if bad > 0 {
            failed.push(format!(
                "{format:?} ({bad} texels):\n  {}",
                examples.join("\n  ")
            ));
        }
    }
    eprintln!("{}", report.join("\n"));
    if !skipped.is_empty() {
        eprintln!("skipped, the adapter lacks them: {}", skipped.join(", "));
    }
    assert!(
        failed.is_empty(),
        "decoded differently from the GPU:\n{}",
        failed.join("\n")
    );
}

/// Texel values that are valid in every format: in range, finite.
fn uncompressed_data(format: TextureFormat, texels: usize, rng: &mut Rng) -> Vec<u8> {
    use TextureFormat::*;
    let n = format.channels();
    let vals: Vec<f32> = (0..texels * n)
        .map(|i| {
            let u = (rng.next() % 100_000) as f32 / 100_000.0;
            match format {
                // HDR ones up to a few thousand, the rest in [0, 1].
                Rgba16Float | R16Float | Rg16Float | R32Float | Rg32Float | Rgba32Float
                | Rg11b10Ufloat | Rgb9e5Ufloat => {
                    if i % 7 == 0 {
                        u * 3000.0
                    } else {
                        u
                    }
                }
                _ => u,
            }
        })
        .collect();
    format.pack_f32(&vals).unwrap()
}

#[test]
fn every_uncompressed_format_reads_back_as_its_texels() {
    let Some(gpu) = device("uncompressed formats", WANTED) else {
        return;
    };
    let features = gpu.device.features();
    let mut rng = Rng(42);
    let (w, h) = (16u32, 16u32);
    let mut failed = Vec::new();
    for format in TextureFormat::all()
        .into_iter()
        .filter(|f| !f.is_block_compressed())
    {
        if format.needs_16bit_norm()
            && !features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            eprintln!("skipping {format:?}: no 16-bit norm");
            continue;
        }
        let data = uncompressed_data(format, (w * h) as usize, &mut rng);
        let cpu = cpu_decode(format, w, h, &data);
        let hw = gpu_decode(&gpu, format, w, h, &data);
        // Exact, but in relative terms: the floats are compared as the f32s
        // both sides computed, the sRGB format through the decode it implies.
        let srgb = format.is_srgb();
        let bad: Vec<String> = cpu
            .iter()
            .zip(&hw)
            .enumerate()
            .filter(|(_, (c, g))| {
                (0..4).any(|k| {
                    let gv = if srgb && k < 3 {
                        srgb_encode(g[k])
                    } else {
                        g[k]
                    };
                    let scale = c[k].abs().max(1.0);
                    (c[k] - gv).abs() / scale > if srgb { 0.6 / 255.0 } else { 1e-6 }
                })
            })
            .take(4)
            .map(|(i, (c, g))| format!("texel {i}: cpu {c:?} gpu {g:?}"))
            .collect();
        if !bad.is_empty() {
            failed.push(format!("{format:?}:\n  {}", bad.join("\n  ")));
        }
    }
    assert!(
        failed.is_empty(),
        "read back differently:\n{}",
        failed.join("\n")
    );
}

/// A GPU without a compression family still shows the texture: the upload
/// decodes it on the CPU, and what the device then samples is what a device
/// with the family samples natively.
#[test]
fn a_gpu_without_a_family_shows_what_one_with_it_does() {
    let Some(native) = device("native", WANTED) else {
        return;
    };
    // Straight from the adapter: the crate's own contexts ask for every
    // texture feature there is.
    let (bare, bare_queue) =
        pollster::block_on(native.adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("no texture features"),
            required_features: wgpu::Features::empty(),
            required_limits: native.adapter.limits(),
            ..Default::default()
        }))
        .expect("a device with no optional features");
    assert!(
        !bare.features().intersects(
            wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
        ),
        "the bare device must lack every family for this to test the fallback"
    );
    let mut rng = Rng(7);
    for format in [
        TextureFormat::Bc7RgbaUnorm,
        TextureFormat::Bc6hRgbUfloat,
        TextureFormat::Bc5RgUnorm,
        TextureFormat::Etc2Rgba8Unorm,
        TextureFormat::Astc {
            block: AstcBlock::B6x5,
            srgb: false,
        },
    ] {
        let (bw, bh) = format.block_dim();
        let (w, h) = (8 * bw, 8 * bh);
        let mut tex = Texture::new(w, h, format, blocks(format, 64, &mut rng));
        tex.flip_y = false;
        if !native
            .device
            .features()
            .contains(match format.compression().unwrap() {
                Compression::Bc => wgpu::Features::TEXTURE_COMPRESSION_BC,
                Compression::Etc2 => wgpu::Features::TEXTURE_COMPRESSION_ETC2,
                Compression::Astc => wgpu::Features::TEXTURE_COMPRESSION_ASTC,
            })
        {
            continue;
        }
        let up = threers::renderer::gpu_texture::GpuTexture::upload(&bare, &bare_queue, &tex);
        assert_eq!(
            up.texture.format(),
            threers::renderer::gpu_texture::wgpu_format(codec::decoded_format(format)),
            "{format:?} on a device without its family"
        );
        let fallback = read_back(&bare, &bare_queue, &up.view, w, h);
        let hw = gpu_decode(&native, format, w, h, &tex.data);
        // The fallback stores 8 bits (or halves); allow that, and the
        // hardware rounding the block formats are allowed.
        let (worst, bad, examples) = compare(format, &hw, &fallback, tolerance(format) + 0.5);
        assert_eq!(
            bad,
            0,
            "{format:?}: worst {worst}\n  {}",
            examples.join("\n  ")
        );
    }
}

/// The same for the uncompressed formats a device can lack: 16-bit unorm
/// without `TEXTURE_FORMAT_16BIT_NORM`, 32-bit float without
/// `FLOAT32_FILTERABLE`. Both become half floats there, and must sample as
/// the native formats do to half-float precision.
#[test]
fn a_gpu_without_16bit_norm_or_float_filtering_still_samples_them() {
    use TextureFormat::*;
    let Some(native) = device("native", WANTED) else {
        return;
    };
    let (bare, bare_queue) =
        pollster::block_on(native.adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("no texture features"),
            required_features: wgpu::Features::empty(),
            required_limits: native.adapter.limits(),
            ..Default::default()
        }))
        .expect("a device with no optional features");
    let mut rng = Rng(3);
    let (w, h) = (16u32, 16u32);
    for format in [R16Unorm, Rgba16Unorm, R32Float, Rg32Float, Rgba32Float] {
        if format.needs_16bit_norm()
            && !native
                .device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            continue;
        }
        let vals: Vec<f32> = (0..(w * h) as usize * format.channels())
            .map(|_| (rng.next() % 65536) as f32 / 65535.0)
            .collect();
        let mut tex = Texture::new(w, h, format, format.pack_f32(&vals).unwrap());
        tex.flip_y = false;
        let up = threers::renderer::gpu_texture::GpuTexture::upload(&bare, &bare_queue, &tex);
        let held = codec::device_format(
            format,
            threers::renderer::gpu_texture::support(bare.features()),
        );
        assert!(
            matches!(held, R16Float | Rg16Float | Rgba16Float),
            "{format:?} held as {held:?}"
        );
        let fallback = read_back(&bare, &bare_queue, &up.view, w, h);
        let hw = gpu_decode(&native, format, w, h, &tex.data);
        for (i, (a, b)) in hw.iter().zip(&fallback).enumerate() {
            for k in 0..4 {
                // A half float keeps 11 bits: within 2^-11 of a value in [0, 1].
                assert!(
                    (a[k] - b[k]).abs() <= 1.0 / 2048.0,
                    "{format:?} texel {i}: {a:?} vs {b:?}"
                );
            }
        }
    }
}

/// What `device_format` picks, for the cases that are policy rather than
/// arithmetic.
#[test]
fn a_device_holds_each_format_in_one_it_can_sample() {
    use TextureFormat::*;
    let none = Support::default();
    let all = Support::ALL;
    assert_eq!(
        codec::device_format(Bc7RgbaUnormSrgb, all),
        Bc7RgbaUnormSrgb
    );
    assert_eq!(codec::device_format(Bc7RgbaUnormSrgb, none), Rgba8UnormSrgb);
    assert_eq!(codec::device_format(Bc4RUnorm, none), R8Unorm);
    assert_eq!(codec::device_format(Bc6hRgbFloat, none), Rgba16Float);
    let astc = Astc {
        block: AstcBlock::B10x8,
        srgb: true,
    };
    assert_eq!(codec::device_format(astc, all), astc);
    assert_eq!(
        codec::device_format(astc, Support { astc: false, ..all }),
        Rgba8UnormSrgb
    );
    assert_eq!(
        codec::device_format(
            R16Unorm,
            Support {
                unorm16: false,
                ..all
            }
        ),
        R32Float
    );
    assert_eq!(codec::device_format(R16Unorm, none), R16Float);
    assert_eq!(codec::device_format(Rgba32Float, none), Rgba16Float);
    assert_eq!(codec::device_format(Rgb9e5Ufloat, none), Rgb9e5Ufloat);
}

/// Every format survives its code (cache keys, the `.tex` blob) and has a
/// consistent sRGB twin.
#[test]
fn codes_are_unique_and_srgb_twins_pair_up() {
    let all = TextureFormat::all();
    let mut codes: Vec<u16> = all.iter().map(|f| f.code()).collect();
    codes.sort();
    codes.dedup();
    assert_eq!(codes.len(), all.len());
    for f in all {
        assert_eq!(TextureFormat::from_code(f.code()), Some(f));
        if let Some(lin) = f.with_srgb(false) {
            assert!(!lin.is_srgb());
            assert_eq!(
                lin.with_srgb(true).map(|s| s.is_srgb()),
                Some(true),
                "{f:?}"
            );
            assert_eq!(lin.bytes_per_block(), f.bytes_per_block());
            assert_eq!(lin.block_dim(), f.block_dim());
        }
    }
}
