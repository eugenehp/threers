//! Averaging a supersampled render down to its output size, on the GPU.
//!
//! Supersampling renders at `size × factor` and resolves each `factor × factor`
//! block down to one pixel. Doing that on the CPU means reading the *large*
//! frame across the bus and then walking it with a transfer function per
//! channel per sample, and it is far more expensive than anything else in a
//! headless frame — at 1920×1080 with `supersample(2)`:
//!
//! ```text
//! render + readback of 3840×2160   4.4 ms
//! CPU average down to 1920×1080   33.1 ms
//! ```
//!
//! Seven times the cost of drawing the frame, to average it. Done here instead,
//! the resolve is part of the frame the GPU is already working on and the
//! readback shrinks by `factor²` — 33 MB to 8 MB — because the copy now starts
//! from the small texture.
//!
//! **Averaging happens in linear light.** sRGB is a transfer function, not a
//! quantity you can take the mean of; averaging encoded bytes directly darkens
//! every edge, and a CAD render or a starfield is mostly edges. So the shader
//! decodes, filters, and re-encodes — matching what the CPU path did, and what
//! [`gpu_texture::MipGenerator`](super::gpu_texture::MipGenerator) gets for free
//! from the texture unit when it has a real sRGB format to work with.
//!
//! We cannot use that free path here: the renderer writes sRGB-*encoded* bytes
//! into a plain `Rgba8Unorm` target (its shader does the encode, see
//! `linear_fb` in `renderer.rs`), so the hardware has no idea the bytes are
//! encoded and would happily average them as-is. A few `pow` calls per output
//! pixel is nothing on a GPU; being wrong is not.
//!
//! # Why the block average is not enough
//!
//! Averaging the `factor × factor` block *is* the pixel's area coverage, which
//! sounds like the right answer and is not. A box one output pixel wide is a
//! poor low-pass: its response is `sinc`, whose first zero sits at twice
//! Nyquist, so it passes a great deal of the detail it was supposed to remove
//! and that detail folds back as aliasing. Adding samples inside the box does
//! not help, because the problem is the box's *support*, not how finely it is
//! sampled — which is why supersampling harder stops paying off so early.
//!
//! Measured on a rotating phone rendered at 8K, resolved 4× to 1080p, tracking
//! the sub-pixel position of one near-horizontal rail edge across eight frames.
//! "Wobble" is how far that edge moves frame to frame *beyond* the rigid motion
//! shared by the whole edge — pure sampling error, and what reads to the eye as
//! an edge that crawls or shimmers rather than glides:
//!
//! ```text
//! filter     taps/axis   wobble    ringing
//! Box            4       0.165 px   none
//! Tent           8       0.097 px   none
//! Mitchell      16       0.086 px   -0.014
//! Lanczos2      16       0.062 px   -0.041
//! ```
//!
//! **If the caller is also accumulating, pick
//! [`Box`](DownsampleFilter::Box) instead.** The resolve filter and the jitter
//! in [`set_taa_converge`](super::Renderer::set_taa_converge) are both
//! reconstruction, and running both is filtering twice. Accumulation already
//! spreads its samples across the whole pixel, which *is* a box reconstruction
//! of it, so the resolve only has to average the block and anything wider is
//! softening for nothing. Measured on a rotating phone at sixteen accumulated
//! samples, with the mean taken at output resolution and in linear light, box
//! against Mitchell against Lanczos-2: rail edge width 1.35 px against 1.55 and
//! 1.50, where 1.0 is a perfect step. Mitchell does measure less wobble along
//! that edge -- 0.55 px against 0.68 -- but that is the blur talking, and it is
//! the same trade a whole-frame blur offered and lost: smoother numbers for
//! detail the render was asked to keep. Without accumulation the ordering is
//! the other way round and by a wider margin, which is why the default stays
//! where it is.
//!
//! [`Mitchell`](DownsampleFilter::Mitchell) is the default: half the sampling
//! error of the box for a filter whose negative lobes are small enough not to
//! ring visibly. Lanczos-2 measures better still, but −0.041 of undershoot on a
//! 100:1 edge is a dark halo you can see, so it is opt-in. Lanczos-3 was tried
//! and is *worse* than the box here — it rings hard enough to displace the very
//! edge it is meant to stabilise.
//!
//! The filters are separable and applied in two passes, so a radius-2 kernel
//! costs `4 × factor` taps per axis rather than `16 × factor²` in one pass. The
//! intermediate is `Rgba16Float`: linear light, and signed, because Mitchell
//! and Lanczos both land outside `[0, 1]` and clamping in between would quietly
//! turn them back into worse versions of the box.

/// The reconstruction filter a [`Downsampler`] resolves with.
///
/// Wider is smoother and costs more taps. See the module docs for what the
/// difference is actually worth; the short version is that
/// [`Box`](Self::Box) is the one to avoid unless you need it for compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DownsampleFilter {
    /// The plain `factor × factor` area average — what
    /// [`average_blocks_srgb`] computes, and the only filter that matches it
    /// bit for bit. Cheapest, and the most aliasing.
    Box,
    /// Triangle, radius one output pixel. Two taps per output pixel per axis
    /// per `factor`, no negative lobes, and most of the box's aliasing gone.
    /// The one to pick when ringing is unacceptable.
    Tent,
    /// Mitchell–Netravali with `B = C = 1/3`, radius two output pixels. The
    /// default: the usual choice for image downscaling, and the best sharpness
    /// available before ringing becomes visible.
    #[default]
    Mitchell,
    /// Windowed sinc, radius two output pixels. The least aliasing measured
    /// here and the most ringing — a visible dark fringe against a bright
    /// edge. Worth it for synthetic detail, not for a product shot.
    Lanczos2,
}

impl DownsampleFilter {
    /// Radius in output pixels. `Box` is special-cased: its support is the
    /// block itself, which is half an output pixel either side of the centre.
    fn radius(self) -> f32 {
        match self {
            Self::Box => 0.5,
            Self::Tent => 1.0,
            Self::Mitchell | Self::Lanczos2 => 2.0,
        }
    }

    /// The kernel at `x` output pixels from the centre.
    fn weight(self, x: f32) -> f32 {
        let x = x.abs();
        match self {
            Self::Box => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Tent => (1.0 - x).max(0.0),
            Self::Mitchell => {
                // Mitchell & Netravali, "Reconstruction Filters in Computer
                // Graphics" (1988), with the B = C = 1/3 they recommend.
                const B: f32 = 1.0 / 3.0;
                const C: f32 = 1.0 / 3.0;
                let (x2, x3) = (x * x, x * x * x);
                if x < 1.0 {
                    ((12.0 - 9.0 * B - 6.0 * C) * x3
                        + (-18.0 + 12.0 * B + 6.0 * C) * x2
                        + (6.0 - 2.0 * B))
                        / 6.0
                } else if x < 2.0 {
                    ((-B - 6.0 * C) * x3
                        + (6.0 * B + 30.0 * C) * x2
                        + (-12.0 * B - 48.0 * C) * x
                        + (8.0 * B + 24.0 * C))
                        / 6.0
                } else {
                    0.0
                }
            }
            Self::Lanczos2 => {
                if x < 1.0e-6 {
                    1.0
                } else if x < 2.0 {
                    let pi_x = std::f32::consts::PI * x;
                    (pi_x.sin() / pi_x) * ((pi_x * 0.5).sin() / (pi_x * 0.5))
                } else {
                    0.0
                }
            }
        }
    }

    /// Weights for one output pixel, and the offset of the first tap from
    /// `output_pixel * factor` in source samples.
    ///
    /// Output pixel `j` covers source samples `[j·f, j·f + f)`, so its centre
    /// sits at `j·f + f/2`; source sample `i` has its centre at `i + 0.5`. Tap
    /// `k` therefore lands `(k + off + 0.5 − f/2) / f` output pixels from the
    /// centre. Getting that half-sample wrong shifts the whole image by half a
    /// pixel, which looks like nothing until it is composited against
    /// something that was resolved correctly.
    fn taps(self, factor: u32) -> (Vec<f32>, i32) {
        let f = factor as f32;
        if self == Self::Box {
            return (vec![1.0 / f; factor as usize], 0);
        }
        // Enough taps to cover the radius either side of the block.
        let half = (self.radius() * f).ceil() as i32;
        let n = 2 * half;
        let off = -half + factor as i32 / 2;
        let mut w: Vec<f32> = (0..n)
            .map(|k| self.weight(((k + off) as f32 + 0.5 - f * 0.5) / f))
            .collect();
        let sum: f32 = w.iter().sum();
        // A degenerate kernel would produce a black frame; fall back to the box
        // rather than divide by nothing. Unreachable for the filters above.
        if sum.abs() < 1.0e-8 {
            return (vec![1.0 / f; factor as usize], 0);
        }
        for v in &mut w {
            *v /= sum;
        }
        (w, off)
    }
}

/// Resolves a supersampled texture down to its output size in one or two
/// render passes.
///
/// Owns its destination texture, sized `src / factor`. Build one per source
/// texture and keep it: everything it needs — pipelines, bind groups,
/// destination, and the intermediate a separable filter needs — is created
/// once, so [`resolve`](Self::resolve) allocates nothing.
///
/// ```no_run
/// # use threers::renderer::downsample::Downsampler;
/// # fn demo(device: &wgpu::Device, queue: &wgpu::Queue, hires: &wgpu::Texture) {
/// let ds = Downsampler::new(device, hires, 2).unwrap();
/// let mut enc = device.create_command_encoder(&Default::default());
/// ds.resolve(&mut enc);
/// queue.submit(Some(enc.finish()));
/// // ds.texture() now holds the resolved frame, ready to copy back.
/// # }
/// ```
pub struct Downsampler {
    factor: u32,
    filter: DownsampleFilter,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    /// Linear-light scratch between the horizontal and vertical passes. `None`
    /// for [`DownsampleFilter::Box`], which resolves in one pass.
    mid: Option<wgpu::Texture>,
    stages: Vec<Stage>,
}

/// One recorded render pass: what it reads, what it draws into.
struct Stage {
    target: wgpu::TextureView,
    bind: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

/// Whether a format's bytes are sRGB-encoded and so must be decoded before
/// they can be averaged.
///
/// One predicate shared with the renderer — see
/// [`shader_applies_transfer`](super::shader_applies_transfer) — because the
/// question is the same one: an sRGB format decodes in hardware when sampled,
/// so decoding again here would be the second half of a double conversion.
fn holds_encoded_srgb(format: wgpu::TextureFormat) -> bool {
    super::shader_applies_transfer(format)
}

impl Downsampler {
    /// Build a downsampler for `src`, resolving `factor × factor` blocks with
    /// the default filter ([`DownsampleFilter::Mitchell`]).
    ///
    /// `None` if `factor` is 1 (there is nothing to do), if the format cannot be
    /// rendered into, or if the source is not an exact multiple of `factor` —
    /// a partial block at the edge would be resolved against whatever the
    /// texture happens to hold there.
    pub fn new(device: &wgpu::Device, src: &wgpu::Texture, factor: u32) -> Option<Self> {
        Self::with_filter(device, src, factor, DownsampleFilter::default())
    }

    /// [`new`](Self::new) with the reconstruction filter spelled out.
    pub fn with_filter(
        device: &wgpu::Device,
        src: &wgpu::Texture,
        factor: u32,
        filter: DownsampleFilter,
    ) -> Option<Self> {
        let format = src.format();
        let (sw, sh) = (src.width(), src.height());
        if factor < 2 || !Self::renderable(format) || sw % factor != 0 || sh % factor != 0 {
            return None;
        }
        let (width, height) = (sw / factor, sh / factor);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers downsample layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    // `textureLoad`, so no filtering is needed — and asking
                    // for none keeps float targets bindable.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers downsample pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        // Both transfer variants, so a later pass can view the same bytes as
        // linear or as sRGB. Bloom needs the sRGB view to get its decode for
        // free, and this texture is what it is handed whenever supersampling
        // is on — which left bloom panicking at view creation on any target
        // that was not already sRGB, i.e. the default one. `RenderTarget` has
        // always declared them; this did not.
        let mut extra_formats = Vec::new();
        for v in [format.add_srgb_suffix(), format.remove_srgb_suffix()] {
            if v != format && !extra_formats.contains(&v) {
                extra_formats.push(v);
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers downsample target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &extra_formats,
        });

        let (taps, off) = filter.taps(factor);
        let decode = holds_encoded_srgb(format);
        let build = |label: &str,
                     source: String,
                     src_view: &wgpu::TextureView,
                     target: wgpu::TextureView,
                     out: wgpu::TextureFormat|
         -> Stage {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{}\n{source}", crate::renderer::fullscreen::VS).into(),
                ),
            });
            Stage {
                target,
                bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers downsample bind"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    }],
                }),
                pipeline: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("threers downsample pipeline"),
                    layout: Some(&pl),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(crate::renderer::fullscreen::VS_ENTRY),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        targets: &[Some(out.into())],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
            }
        };

        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let (mid, stages) = if filter == DownsampleFilter::Box {
            // One pass over the block, which is what `average_blocks_srgb`
            // computes — and the only arrangement that matches it bit for bit,
            // since resolving separably rounds twice.
            let stage = build(
                "threers downsample (box)",
                box_shader(factor, decode),
                &src_view,
                texture.create_view(&wgpu::TextureViewDescriptor::default()),
                format,
            );
            (None, vec![stage])
        } else {
            // Signed and linear: Mitchell and Lanczos both undershoot, and
            // clamping between the passes would turn them into worse boxes.
            let mid = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers downsample scratch"),
                size: wgpu::Extent3d {
                    width,
                    height: sh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let mid_view = mid.create_view(&wgpu::TextureViewDescriptor::default());
            let h = build(
                "threers downsample (horizontal)",
                separable_shader(&taps, off, factor, Axis::X, decode, false),
                &src_view,
                mid.create_view(&wgpu::TextureViewDescriptor::default()),
                wgpu::TextureFormat::Rgba16Float,
            );
            let v = build(
                "threers downsample (vertical)",
                separable_shader(&taps, off, factor, Axis::Y, false, decode),
                &mid_view,
                texture.create_view(&wgpu::TextureViewDescriptor::default()),
                format,
            );
            (Some(mid), vec![h, v])
        };

        Some(Self {
            factor,
            filter,
            width,
            height,
            texture,
            mid,
            stages,
        })
    }

    /// Formats that can be both loaded from and rendered into.
    fn renderable(format: wgpu::TextureFormat) -> bool {
        matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Rgba16Float
        )
    }

    /// Record the resolve. The result lands in [`texture`](Self::texture).
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        for stage in &self.stages {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers downsample pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &stage.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Every pixel is written, so there is nothing to preserve.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// The resolved texture, at [`size`](Self::size).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
    /// Output size — the source size divided by [`factor`](Self::factor).
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// The block size being resolved.
    pub fn factor(&self) -> u32 {
        self.factor
    }
    /// The reconstruction filter in use.
    pub fn filter(&self) -> DownsampleFilter {
        self.filter
    }
    /// Bytes of GPU memory held beyond the output texture — the separable
    /// filters' linear-light scratch, and zero for [`DownsampleFilter::Box`].
    pub fn scratch_bytes(&self) -> u64 {
        match &self.mid {
            Some(t) => t.width() as u64 * t.height() as u64 * 8,
            None => 0,
        }
    }
}

/// Which axis a separable pass filters along.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// The bits every pass shares: the fullscreen triangle and the two transfer
/// functions.
fn preamble() -> &'static str {
    r#"
@group(0) @binding(0) var src : texture_2d<f32>;

// One oversized triangle rather than two: no seam down the diagonal.
fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)),
                  c / 12.92,
                  c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055,
                  c * 12.92,
                  c <= vec3<f32>(0.0031308));
}
"#
}

/// `textureLoad` with the coordinate clamped and the colour decoded.
///
/// Clamping rather than bounds-checking: a radius-2 kernel reaches two output
/// pixels past the frame, and WGSL leaves an out-of-range `textureLoad`
/// indeterminate. Replicating the edge pixel is the standard answer and keeps
/// the weights summing to one, so the border neither darkens nor brightens.
fn tap_fn(decode: bool) -> String {
    let d = if decode {
        "srgb_to_linear(c.rgb)"
    } else {
        "c.rgb"
    };
    format!(
        "
fn tap(at : vec2<i32>, dims : vec2<i32>) -> vec4<f32> {{
    let c = textureLoad(src, clamp(at, vec2<i32>(0), dims - vec2<i32>(1)), 0);
    return vec4<f32>({d}, c.a);
}}
"
    )
}

/// How a pass finishes: negatives clipped, and re-encoded if the target holds
/// encoded sRGB.
///
/// Mitchell and Lanczos undershoot, and negative light is not a thing — an
/// 8-bit target would clamp it anyway, so clamping here keeps every format
/// agreeing. The overshoot on the bright side is left alone, which is where the
/// sharpness comes from.
fn finish(encode: bool) -> &'static str {
    if encode {
        "    let rgb = clamp(acc.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(linear_to_srgb(rgb), clamp(acc.a, 0.0, 1.0));"
    } else {
        "    return vec4<f32>(max(acc.rgb, vec3<f32>(0.0)), clamp(acc.a, 0.0, 1.0));"
    }
}

/// The one-pass `factor × factor` average — [`DownsampleFilter::Box`].
fn box_shader(factor: u32, srgb: bool) -> String {
    format!(
        "{}{}
@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {{
    let dims = vec2<i32>(textureDimensions(src));
    // `pos` is the fragment centre, so truncating gives the output pixel.
    let base = vec2<i32>(pos.xy) * {factor};
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < {factor}; y = y + 1) {{
        for (var x = 0; x < {factor}; x = x + 1) {{
            acc = acc + tap(base + vec2<i32>(x, y), dims);
        }}
    }}
    acc = acc / {n:?};
{}
}}
",
        preamble(),
        tap_fn(srgb),
        finish(srgb),
        n = (factor * factor) as f32,
    )
}

/// One axis of a separable filter, with the weights unrolled into the source.
///
/// Unrolled rather than looped over a weights buffer: the tap count is known
/// when the pipeline is built, the compiler gets to fold the arithmetic, and
/// there is no uniform to bind or keep in step.
fn separable_shader(
    taps: &[f32],
    off: i32,
    factor: u32,
    axis: Axis,
    decode: bool,
    encode: bool,
) -> String {
    let mut body = String::new();
    for (k, w) in taps.iter().enumerate() {
        let d = k as i32 + off;
        let at = match axis {
            Axis::X => format!("vec2<i32>(base + ({d}), o.y)"),
            Axis::Y => format!("vec2<i32>(o.x, base + ({d}))"),
        };
        body.push_str(&format!("    acc = acc + tap({at}, dims) * {w:?};\n"));
    }
    let base = match axis {
        Axis::X => format!("o.x * {factor}"),
        Axis::Y => format!("o.y * {factor}"),
    };
    format!(
        "{}{}
@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {{
    let dims = vec2<i32>(textureDimensions(src));
    let o = vec2<i32>(pos.xy);
    let base = {base};
    var acc = vec4<f32>(0.0);
{body}{}
}}
",
        preamble(),
        tap_fn(decode),
        finish(encode),
    )
}

/// Average `factor × factor` blocks of RGBA8 down to one pixel, on the CPU.
///
/// The reference the GPU [`DownsampleFilter::Box`] pass is tested against, and
/// the fallback for the cases [`Downsampler::new`] rejects. Prefer
/// [`HeadlessRenderer::render_to_rgba_resolved`](super::HeadlessRenderer::render_to_rgba_resolved),
/// which is about fifteen times faster and resolves with a better filter; this
/// is here so the contract holds even where the GPU path cannot.
///
/// Colour is averaged in linear light, alpha as it stands — alpha carries no
/// transfer function.
pub fn average_blocks_srgb(src: &[u8], w: u32, h: u32, factor: u32) -> Vec<u8> {
    let (dw, dh) = (w / factor, h / factor);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let n = (factor * factor) as f32;
    let to_linear = |b: u8| {
        let s = b as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_srgb = |l: f32| {
        let s = if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = [0.0f32; 4];
            for sy in 0..factor {
                for sx in 0..factor {
                    let i = (((y * factor + sy) * w + (x * factor + sx)) * 4) as usize;
                    acc[0] += to_linear(src[i]);
                    acc[1] += to_linear(src[i + 1]);
                    acc[2] += to_linear(src[i + 2]);
                    acc[3] += src[i + 3] as f32 / 255.0;
                }
            }
            let o = ((y * dw + x) * 4) as usize;
            out[o] = to_srgb(acc[0] / n);
            out[o + 1] = to_srgb(acc[1] / n);
            out[o + 2] = to_srgb(acc[2] / n);
            out[o + 3] = ((acc[3] / n).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}
