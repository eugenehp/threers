//! Drop the alpha channel on the GPU, before the frame is ever read back.
//!
//! A rendered frame is RGBA because that is what a colour attachment is. A
//! video frame is not: every encoder here wants three bytes a pixel, so the
//! fourth is carried across the PCIe/unified-memory readback, handed to a video
//! encoder and thrown away. At 8K that is 33 MB a frame of pure waste on the
//! way out, and the same again on the way in to swscale.
//!
//! Doing the pack on the CPU instead is worse than doing nothing, which is not
//! obvious until it is measured. One pass over 33 Mpx copying three bytes at a
//! time vectorises badly, and threaded across the cores it saturates the memory
//! bandwidth the render loop needs for its own readback: an 8K frame went from
//! 37.7 ms in output to 236.5, and readback from 3.4 ms to 36.8 — 232 ms a
//! frame spent to save the encoder 150.
//!
//! # Why four pixels per invocation
//!
//! WGSL storage buffers are addressed in `u32`s, so writing three bytes per
//! pixel would mean read-modify-write across word boundaries, and neighbouring
//! invocations would collide on the shared word. Four pixels is the smallest
//! group whose packed form — 12 bytes — is a whole number of words. So each
//! invocation reads 4 texels and writes exactly 3 words that nobody else
//! touches: no atomics, no barriers, no overlap.
//!
//! The tail is handled by clamping reads to the last pixel rather than by a
//! branch: a frame whose pixel count is not a multiple of four writes a few
//! bytes of padding past the end, which `read` trims off.

/// A compute pass that packs an RGBA texture into tightly-packed RGB bytes.
pub struct RgbPack {
    /// Kept so the reprojection matrices can be refreshed each frame; they
    /// change every frame and a pack built once would blur along stale ones.
    ubo: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// The packed bytes, as a storage buffer the readback copies from.
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    /// Bytes actually meaningful: `width * height * 3`.
    len: usize,
    groups_x: u32,
    groups_y: u32,
}

const WORKGROUP: u32 = 64;

/// Vignette and grain, as a pair, because they are the two things that say
/// "photograph" rather than "render" and neither is part of the scene.
///
/// Zero on both is bit-for-bit the frame without them -- the shader branches
/// out entirely -- so this is free to leave in the pipeline and off by default.
#[derive(Clone, Copy, Debug, Default)]
pub struct FilmGrade {
    /// 0 none, 1 a full cos-fourth falloff. 0.2-0.4 is a lens; 1.0 is a look.
    pub vignette: f32,
    /// Peak grain amplitude in the mid-tones, in output units. 0.02 is 35 mm.
    pub grain: f32,
    /// Frame number, so the grain moves. Anything that changes per frame.
    pub seed: f32,
}

impl RgbPack {
    /// Build the pass for `src`, or `None` if its format cannot be sampled as
    /// plain 8-bit colour — the HDR path has to stay on the float readback.
    pub fn new(device: &wgpu::Device, src: &wgpu::Texture) -> Option<Self> {
        Self::with_bloom(device, src, None, 0.0)
    }

    /// As [`new`](Self::new), but adds `bloom` (a lower-resolution linear-light
    /// texture) scaled by `strength` before packing.
    ///
    /// Compositing HERE rather than in a pass of its own is what keeps the
    /// frame from being touched twice: this shader already reads every pixel on
    /// its way to the encoder, so the add is free beyond the sample. With
    /// `strength` at zero the output is bit-for-bit what it was without bloom,
    /// which is what lets the byte-exactness test keep meaning something.
    pub fn with_bloom(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        bloom: Option<&wgpu::TextureView>,
        strength: f32,
    ) -> Option<Self> {
        Self::with_post(device, src, bloom, strength, None)
    }

    /// As [`with_bloom`](Self::with_bloom), plus an ambient-occlusion factor
    /// multiplied into the frame. Both are composited in this one pass because
    /// it already reads every pixel; adding either as its own pass over the
    /// picture would cost more than the effects do.
    pub fn with_post(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        bloom: Option<&wgpu::TextureView>,
        strength: f32,
        ao: Option<&wgpu::TextureView>,
    ) -> Option<Self> {
        Self::with_post_mb(
            device, src, bloom, strength, ao, None, [0.0; 16], [0.0; 16], 0.0, None,
        )
    }

    /// As [`with_post_mb`](Self::with_post_mb), plus the film grade.
    ///
    /// Vignette and grain belong in THIS pass and not on the CPU, for the same
    /// reason the masthead does: the frame is read back straight from the buffer
    /// the GPU wrote it to, so touching it afterwards costs a full copy of every
    /// frame. Here they are two lines in a shader that already reads every pixel.
    ///
    /// Both go on BEFORE the overlay, so the masthead and the captions stay
    /// clean -- they are graphics laid on the picture, and a vignette across a
    /// logo strip reads as a mistake.
    #[allow(clippy::too_many_arguments)]
    pub fn with_post_film(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        bloom: Option<&wgpu::TextureView>,
        strength: f32,
        ao: Option<&wgpu::TextureView>,
        depth: Option<&wgpu::Texture>,
        inv_vp: [f32; 16],
        prev_vp: [f32; 16],
        shutter: f32,
        overlay: Option<&wgpu::Texture>,
        film: FilmGrade,
    ) -> Option<Self> {
        Self::build(
            device, src, bloom, strength, ao, depth, inv_vp, prev_vp, shutter, overlay, film,
        )
    }

    /// As [`with_post`](Self::with_post), plus camera motion blur.
    ///
    /// `depth` with the two matrices gives each pixel a velocity: reconstruct
    /// where it is, ask where it WAS through the previous camera, and the screen
    /// distance between the two is how far it travelled during the frame.
    /// `shutter` scales that — 0.5 is a 180-degree shutter, the film default,
    /// because a real shutter is open for half the frame interval and not all
    /// of it.
    ///
    /// Camera motion only. Object motion needs per-object velocities, which
    /// means a velocity buffer this renderer does not write; the arm swinging
    /// against a still camera will not blur. For this film the dominant motion
    /// IS the camera, so that is most of the benefit for none of the plumbing.
    #[allow(clippy::too_many_arguments)]
    pub fn with_post_mb(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        bloom: Option<&wgpu::TextureView>,
        strength: f32,
        ao: Option<&wgpu::TextureView>,
        depth: Option<&wgpu::Texture>,
        inv_vp: [f32; 16],
        prev_vp: [f32; 16],
        shutter: f32,
        overlay: Option<&wgpu::Texture>,
    ) -> Option<Self> {
        Self::build(
            device, src, bloom, strength, ao, depth, inv_vp, prev_vp, shutter, overlay,
            FilmGrade::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        bloom: Option<&wgpu::TextureView>,
        strength: f32,
        ao: Option<&wgpu::TextureView>,
        depth: Option<&wgpu::Texture>,
        inv_vp: [f32; 16],
        prev_vp: [f32; 16],
        shutter: f32,
        // An RGBA overlay drawn over the top of the frame, anchored at the top
        // left — a masthead, a lower third, anything the caller composites.
        //
        // It belongs HERE and not on the CPU. Compositing it into the frame
        // after readback means the frame has to be WRITABLE, which means it has
        // to be copied out of the buffer the GPU wrote it to — 99.5 MB a frame
        // at 8K, purely so a strip across the top can be blended in. Done here
        // the frame is never touched by the CPU at all.
        overlay: Option<&wgpu::Texture>,
        film: FilmGrade,
    ) -> Option<Self> {
        let (width, height) = (src.width(), src.height());
        // textureLoad on a non-filterable float view is the cheapest way in and
        // needs no sampler. Formats outside this set are not what the video
        // path produces.
        let bgra = match src.format() {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            _ => return None,
        };

        let px = width as usize * height as usize;
        let len = px * 3;
        // Round up to whole 4-pixel groups so the last invocation's three words
        // land inside the allocation.
        let words = px.div_ceil(4) * 3;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers rgb pack"),
            size: (words * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // An sRGB view would decode on load and undo the very encoding the file
        // is supposed to carry, so read the texture as UNORM and pass the bytes
        // through untouched.
        let view_format = if bgra {
            wgpu::TextureFormat::Bgra8Unorm
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let view = src.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rgb pack src"),
            format: Some(view_format),
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers rgb pack"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers rgb pack layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // A dispatch dimension caps at 65535 workgroups, and 8K needs 129600 of
        // them, so the grid is 2D and the shader rebuilds the linear index from
        // both axes. `stride` is how many quads one row of the grid covers.
        let quads = px.div_ceil(4) as u32;
        let total_groups = quads.div_ceil(WORKGROUP);
        let groups_x = total_groups.clamp(1, 32768);
        let groups_y = total_groups.div_ceil(groups_x);
        let stride = groups_x * WORKGROUP;
        // width, pixel count, BGRA flag, and the grid stride in quads.
        let params = [width, px as u32, u32::from(bgra), stride];
        let mb_on = depth.is_some() && shutter > 0.0;
        let (ov_w, ov_h) = overlay.map(|t| (t.width(), t.height())).unwrap_or((0, 0));
        let fparams = [
            strength,
            if bloom.is_some() { 1.0f32 } else { 0.0 },
            if ao.is_some() { 1.0f32 } else { 0.0 },
            if mb_on { shutter } else { 0.0 },
        ];
        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers rgb pack params"),
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut m = ubo.slice(..).get_mapped_range_mut().expect("buffer range is mapped");
            m.slice(..16).copy_from_slice(bytemuck::cast_slice(&params));
            m.slice(16..32).copy_from_slice(bytemuck::cast_slice(&fparams));
            m.slice(32..96).copy_from_slice(bytemuck::cast_slice(&inv_vp));
            m.slice(96..160).copy_from_slice(bytemuck::cast_slice(&prev_vp));
            m.slice(160..168).copy_from_slice(bytemuck::cast_slice(&[ov_w, ov_h]));
            m.slice(168..176).copy_from_slice(bytemuck::cast_slice(&[film.vignette, film.grain]));
            m.slice(176..192)
                .copy_from_slice(bytemuck::cast_slice(&[film.seed, 0.0f32, 0.0f32, 0.0f32]));
        }
        ubo.unmap();

        // A 1x1 stand-in when there is no bloom: the binding must be filled
        // either way, and a branch in the shader on a uniform is cheaper than
        // two pipelines.
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgb pack no-bloom"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&Default::default());
        let bloom_view = bloom.unwrap_or(&dummy_view);
        let ao_dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgb pack no-ao"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ao_dummy_view = ao_dummy.create_view(&Default::default());
        let ao_view = ao.unwrap_or(&ao_dummy_view);
        let depth_dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgb pack no-depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ov_dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgb pack no-overlay"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ov_view = match overlay {
            Some(t) => t.create_view(&Default::default()),
            None => ov_dummy.create_view(&Default::default()),
        };
        let depth_view = match depth.filter(|d| d.sample_count() == 1) {
            Some(d) => d.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers rgb pack depth"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            }),
            None => depth_dummy.create_view(&Default::default()),
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers rgb pack bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&ov_view),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers rgb pack pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("threers rgb pack pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            ubo,
            pipeline,
            bind_group,
            buffer,
            width,
            height,
            len,
            groups_x,
            groups_y,
        })
    }

    /// Refresh the reprojection for this frame. Must be called before
    /// [`pack`](Self::pack) on every frame motion blur is wanted on: the
    /// matrices are per-frame, and stale ones smear along last frame's motion.
    pub fn set_reprojection(&self, queue: &wgpu::Queue, inv_vp: [f32; 16], prev_vp: [f32; 16]) {
        queue.write_buffer(&self.ubo, 32, bytemuck::cast_slice(&inv_vp));
        queue.write_buffer(&self.ubo, 96, bytemuck::cast_slice(&prev_vp));
    }

    /// Advance the grain seed. Cheap: eight bytes into a buffer that is already
    /// written every frame, and no pipeline rebuild -- which is the whole reason
    /// the seed is a uniform and not baked in at construction.
    pub fn set_grain_seed(&self, queue: &wgpu::Queue, seed: f32) {
        queue.write_buffer(&self.ubo, 176, bytemuck::cast_slice(&[seed]));
    }

    /// Record the pack. Must run after the frame is drawn and before the copy.
    pub fn pack(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("threers rgb pack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.groups_x, self.groups_y, 1);
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Meaningful byte count — `width * height * 3`, without the group padding.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

const SHADER: &str = r#"
struct Params {
    width    : u32,
    count    : u32,
    bgra     : u32,
    stride   : u32,
    strength : f32,
    has_bloom: f32,
    has_ao   : f32,
    shutter  : f32,
    inv_vp   : mat4x4<f32>,
    prev_vp  : mat4x4<f32>,
    ov_w     : u32,
    ov_h     : u32,
    vig      : f32,
    grain    : f32,
    seed     : f32,
    fpad1    : f32,
    fpad2    : f32,
    fpad3    : f32,
};

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;
@group(0) @binding(2) var<uniform> p : Params;
@group(0) @binding(3) var bloom : texture_2d<f32>;
@group(0) @binding(4) var ao    : texture_2d<f32>;
@group(0) @binding(5) var mb_depth : texture_depth_2d;
@group(0) @binding(6) var overlay : texture_2d<f32>;

// sRGB transfer, both ways. Bloom is gathered and blurred in linear light
// because that is the only space where "how much light is here" is additive;
// the stored frame is sRGB-encoded, so it has to be decoded before the add and
// re-encoded after. Adding encoded values instead brightens midtones far more
// than highlights, which is the opposite of what a bloom should do.
fn to_linear(c : f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}
fn to_srgb(c : f32) -> f32 {
    if (c <= 0.0031308) { return c * 12.92; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// Where this pixel was a frame ago, in UV. Reconstruct its world position from
// depth and look it up through the previous camera; the gap between the two is
// the distance it swept while the shutter was open.
fn velocity(xy : vec2<u32>, dim : vec2<u32>) -> vec2<f32> {
    let d = textureLoad(mb_depth, vec2<i32>(xy), 0);
    if (d >= 1.0) { return vec2<f32>(0.0); }
    let uv = (vec2<f32>(xy) + 0.5) / vec2<f32>(dim);
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let wh = p.inv_vp * ndc;
    let world = wh.xyz / wh.w;
    let pc = p.prev_vp * vec4<f32>(world, 1.0);
    if (pc.w <= 0.0) { return vec2<f32>(0.0); }
    let prev_uv = (pc.xy / pc.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return (uv - prev_uv) * p.shutter;
}

fn texel(i : u32) -> vec3<u32> {
    // Clamp rather than branch: the tail of the last group re-reads the final
    // pixel, and the bytes it writes are past `len` and trimmed on the way out.
    let j = min(i, p.count - 1u);
    let xy = vec2<u32>(j % p.width, j / p.width);
    let c = textureLoad(src, vec2<i32>(xy), 0);
    var rgb = select(c.rgb, c.bgr, p.bgra != 0u);
    if (p.shutter > 0.0) {
        let dim = textureDimensions(src);
        let v = velocity(xy, dim);
        // Below a pixel of travel there is nothing to smear, and sampling
        // anyway would just soften a still frame.
        let px = length(v * vec2<f32>(dim));
        if (px > 1.0) {
            // 32 TAPS, AND A DITHERED PHASE.
            //
            // At 12 the tap COUNT was capped but the tap SPACING was not, so a
            // 120-pixel sweep sampled 12 points ten pixels apart: not a blur,
            // twelve discrete copies. On a big smooth bright subject -- Earth's
            // limb, the Moon's disc -- those copies read as a ladder of edges,
            // which is the "lines when we move fast" this has been showing.
            // Point sources are worse still: a star becomes twelve stars.
            //
            // Raising the cap alone only moves the threshold. What removes the
            // banding is breaking the phase per pixel, so neighbouring pixels
            // sample at different offsets along the streak and the residual
            // undersampling turns into noise instead of a fixed pattern -- and
            // noise at this amplitude is indistinguishable from the blur it is
            // standing in for.
            // TAPS ARE PRICED, NOT MAXED. min(px, 32) sampled one tap per
            // pixel of streak, so any sweep over 32 px paid the full 32 -- and
            // at 8K almost every moving pixel does. Measured on the deck film
            // at 854x480 that put motion blur at 43% of frame time; the cost
            // scales with pixels x taps, so at 8K it dominates.
            //
            // One tap per pixel was only ever needed because the taps landed on
            // a fixed ladder. With the phase dithered per pixel the residual is
            // noise, and noise at one tap per THREE pixels is indistinguishable
            // from the blur it stands in for -- neighbouring pixels fill each
            // other's gaps. So price the taps off the streak and cap at 24:
            //
            //     px  12 ->  6 taps  (was 12)
            //     px  30 -> 12 taps  (was 30)
            //     px 300 -> 24 taps  (was 32)
            let n = clamp(u32(px / 3.0 + 2.0), 4u, 24u);
            var jh = xy.x * 1973u + xy.y * 9277u + u32(p.seed) * 26699u;
            jh = jh ^ (jh >> 15u);
            jh = jh * 2246822519u;
            let jit = f32((jh ^ (jh >> 13u)) & 0xffffu) / 65536.0;
            var acc = rgb;
            var w = 1.0;
            for (var k = 1u; k <= n; k = k + 1u) {
                // (k - jit) not k: the offset is per pixel, so the taps of
                // adjacent pixels interleave instead of landing on the same
                // ladder. jit is in [0,1), so t stays in (0, 1].
                let t = (f32(k) - jit) / f32(n);
                let s = vec2<f32>(xy) - v * vec2<f32>(dim) * t;
                let sc = clamp(vec2<i32>(s), vec2<i32>(0), vec2<i32>(dim) - 1);
                let sm = textureLoad(src, sc, 0);
                acc = acc + select(sm.rgb, sm.bgr, p.bgra != 0u);
                w = w + 1.0;
            }
            rgb = acc / w;
        }
    }
    if (p.has_ao > 0.5) {
        // Occlusion multiplies LIGHT, so it belongs in linear space like the
        // bloom add. Applied to the encoded value it would crush shadows and
        // barely touch highlights.
        let ad = textureDimensions(ao);
        let au = vec2<u32>(min(xy.x * ad.x / max(p.width, 1u), ad.x - 1u),
                           min(xy.y * ad.y / max(textureDimensions(src).y, 1u), ad.y - 1u));
        let k = textureLoad(ao, vec2<i32>(au), 0).r;
        let lin = vec3<f32>(to_linear(rgb.r), to_linear(rgb.g), to_linear(rgb.b)) * k;
        rgb = vec3<f32>(to_srgb(lin.r), to_srgb(lin.g), to_srgb(lin.b));
    }
    if (p.has_bloom > 0.5) {
        // Nearest-tap upsample from the quarter-resolution bloom. It is a blur;
        // there is nothing in it that a bilinear fetch would preserve.
        let bd = textureDimensions(bloom);
        let uv = vec2<u32>(
            min(xy.x * bd.x / max(p.width, 1u), bd.x - 1u),
            min(xy.y * bd.y / max(textureDimensions(src).y, 1u), bd.y - 1u));
        let add = textureLoad(bloom, vec2<i32>(uv), 0).rgb * p.strength;
        let lin = vec3<f32>(to_linear(rgb.r), to_linear(rgb.g), to_linear(rgb.b)) + add;
        rgb = vec3<f32>(to_srgb(lin.r), to_srgb(lin.g), to_srgb(lin.b));
    }
    // THE FILM GRADE, before the overlay for the reason given on with_post_film.
    if (p.vig > 0.0 || p.grain > 0.0) {
        let dim = vec2<f32>(textureDimensions(src));
        if (p.vig > 0.0) {
            // cos-fourth falloff -- the one a real lens has. Normalised so the
            // centre is untouched and only the corners lose light. Vignetting
            // is lost LIGHT, so it multiplies in linear like the AO term.
            let q = (vec2<f32>(xy) + 0.5) / dim * 2.0 - 1.0;
            let f = 1.0 / (1.0 + dot(q, q) * 0.5);
            let k = mix(1.0, f * f, p.vig);
            let lin = vec3<f32>(to_linear(rgb.r), to_linear(rgb.g), to_linear(rgb.b)) * k;
            rgb = vec3<f32>(to_srgb(lin.r), to_srgb(lin.g), to_srgb(lin.b));
        }
        if (p.grain > 0.0) {
            // Hashed on the pixel AND the frame, so it moves: static grain is
            // dirt on the lens, not grain. Weighted to the mid-tones, because
            // film has almost none in the toe or the shoulder and an even
            // sprinkle over black reads as sensor noise.
            var hsh = xy.x * 374761393u + xy.y * 668265263u
                    + u32(p.seed) * 1274126177u;
            hsh = hsh ^ (hsh >> 13u);
            hsh = hsh * 1274126177u;
            let n = f32((hsh ^ (hsh >> 16u)) & 0xffffu) / 65535.0 - 0.5;
            let l = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
            rgb = clamp(rgb + n * p.grain * (4.0 * l * (1.0 - l)),
                        vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }
    // The overlay goes on LAST, over bloom and everything else: it is graphics
    // laid on the picture, not part of the scene, so nothing should bloom it or
    // blur it. Straight alpha, in the frame's own encoding.
    if (p.ov_w > 0u && xy.y < p.ov_h && xy.x < p.ov_w) {
        let o = textureLoad(overlay, vec2<i32>(vec2<u32>(xy.x, xy.y)), 0);
        rgb = mix(rgb, o.rgb, o.a);
    }
    // Otherwise the attachment is already in the encoding the file will carry,
    // so round the stored value and do not touch its transfer function.
    return vec3<u32>(clamp(rgb * 255.0 + 0.5, vec3<f32>(0.0), vec3<f32>(255.0)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let quad = gid.x + gid.y * p.stride;
    let first = quad * 4u;
    if (first >= p.count) { return; }

    let a = texel(first);
    let b = texel(first + 1u);
    let c = texel(first + 2u);
    let d = texel(first + 3u);

    // 4 pixels -> 12 bytes -> exactly 3 words, little-endian, no neighbour
    // shares a word with us.
    let o = quad * 3u;
    dst[o]      = a.x | (a.y << 8u) | (a.z << 16u) | (b.x << 24u);
    dst[o + 1u] = b.y | (b.z << 8u) | (c.x << 16u) | (c.y << 24u);
    dst[o + 2u] = c.z | (d.x << 8u) | (d.y << 16u) | (d.z << 24u);
}
"#;
