//! The pieces every screen-space effect in this crate was writing for itself.
//!
//! A fullscreen pass is always the same shape: a triangle twice the size of the
//! viewport, a float target or a pair of them to ping-pong between, a sampler,
//! a bind group, and a render pass that draws three vertices. Bloom, ambient
//! occlusion, the supersample resolve, the caption overlay, the texture mip
//! builder, PMREM, the post-process composer and the fluid each grew their own
//! copy of that — twelve files with their own fullscreen vertex stage at the
//! last count.
//!
//! This is the shared one. Nothing here is specific to any effect:
//!
//! ```no_run
//! use threers::renderer::fullscreen::{self, Fbo, Load};
//! # fn demo(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder,
//! #         pipeline: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup) {
//! let target = Fbo::new(device, "blur", 512, 512, wgpu::TextureFormat::Rgba16Float);
//! fullscreen::draw(encoder, &target.view, Load::Clear, pipeline, bind_group);
//! # }
//! ```
//!
//! The vertex stage is [`VS`], a WGSL string to prepend to a fragment shader —
//! three vertices, no vertex buffer, no index buffer, `uv` interpolated with
//! the texture's own origin at the top left.

use std::collections::HashMap;

/// A fullscreen triangle, as WGSL to prepend to a fragment shader.
///
/// One triangle rather than a quad's two: the corners outside the viewport
/// clip, and a single primitive means no seam down the diagonal where two
/// triangles meet — which matters for any pass whose derivatives are not
/// continuous across it.
///
/// Declares `vs_main` and a `FullscreenVertex` struct with `uv` at
/// `@location(0)`. Draw it with `pass.draw(0..3, 0..1)`, or with [`draw`].
pub const VS: &str = r#"
struct FullscreenVertex {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> FullscreenVertex {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out : FullscreenVertex;
    out.uv = vec2<f32>(x, y);
    out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}
"#;

/// The entry point [`VS`] declares.
pub const VS_ENTRY: &str = "vs_main";

/// Load the target's existing contents, or throw them away.
///
/// `Clear` is the cheap answer and the right one whenever the pass covers the
/// target, which a fullscreen pass does by definition — so the exception is a
/// pass that *blends*, where the existing contents are half the result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Load {
    Clear,
    Keep,
}

/// A single colour target, plus the view everything binds it through.
///
/// Carries its size, because almost every pass needs `1 / resolution` and
/// deriving it from the texture at each use is how a pass ends up sampling at
/// the wrong scale after a resize.
/// Hands out [`Fbo::id`]. Monotonic and never reused, so a stale cache entry
/// cannot be matched by a later target that happens to land in the same
/// place.
fn next_fbo_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

pub struct Fbo {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// Distinct for every target ever made, so a cache can be keyed on which
    /// textures a bind group was built from.
    ///
    /// The address of the `Fbo` will not do: a [`DoubleFbo`] swaps its two by
    /// moving them, so the same address holds a different texture on
    /// alternate frames — which is exactly the case a cache has to tell
    /// apart.
    pub id: u64,
}

impl Fbo {
    /// A target that can be rendered into, sampled, and copied out of.
    ///
    /// `COPY_SRC` is unconditional on purpose: an intermediate buffer that
    /// cannot be read back is one whose contents can only be judged by eye,
    /// and the cost of the flag on a target that is never copied is nothing.
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
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
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            id: next_fbo_id(),
            texture,
            view,
            width,
            height,
        }
    }

    /// 1 / resolution — the step from one texel to the next, in uv.
    pub fn texel(&self) -> [f32; 2] {
        [1.0 / self.width as f32, 1.0 / self.height as f32]
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }

    /// This target as something to render into. See [`Target`].
    pub fn target(&self) -> Target<'_> {
        Target {
            view: &self.view,
            format: self.texture.format(),
            width: self.width,
            height: self.height,
        }
    }
}

/// A field that is read and written in the same pass, so it needs two copies.
///
/// [`swap`](Self::swap) after each pass that writes it; `read` is always the
/// current contents and `write` always the scratch.
pub struct DoubleFbo {
    pub read: Fbo,
    pub write: Fbo,
}

impl DoubleFbo {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            read: Fbo::new(device, label, width, height, format),
            write: Fbo::new(device, label, width, height, format),
        }
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.read, &mut self.write);
    }

    pub fn width(&self) -> u32 {
        self.read.width
    }

    pub fn height(&self) -> u32 {
        self.read.height
    }

    pub fn size(&self) -> (u32, u32) {
        self.read.size()
    }

    pub fn texel(&self) -> [f32; 2] {
        self.read.texel()
    }
}

/// Somewhere to draw: a view, and the two things a view does not carry.
///
/// `wgpu::TextureView` knows neither its format nor its size, so every API
/// that takes one ends up taking three arguments that have to agree — and the
/// way they stop agreeing is a resize that updates two of them. This keeps
/// them together, and [`Target::from_texture`] derives all three from the one
/// place that has them.
#[derive(Clone, Copy)]
pub struct Target<'a> {
    pub view: &'a wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl<'a> Target<'a> {
    /// A view of a whole texture, with the format and size taken from it.
    ///
    /// The view is returned alongside because it has to outlive the `Target`,
    /// and `create_view` makes a new one each call:
    ///
    /// ```no_run
    /// # use threers::renderer::fullscreen::Target;
    /// # fn demo(texture: &wgpu::Texture) {
    /// let view = texture.create_view(&Default::default());
    /// let target = Target::from_texture(texture, &view);
    /// # let _ = target;
    /// # }
    /// ```
    pub fn from_texture(texture: &wgpu::Texture, view: &'a wgpu::TextureView) -> Self {
        Self {
            view,
            format: texture.format(),
            width: texture.width(),
            height: texture.height(),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// 1 / resolution, in uv.
    pub fn texel(&self) -> [f32; 2] {
        [
            1.0 / self.width.max(1) as f32,
            1.0 / self.height.max(1) as f32,
        ]
    }
}

impl<'a> From<&'a super::RenderTarget> for Target<'a> {
    fn from(rt: &'a super::RenderTarget) -> Self {
        Self {
            view: &rt.color_view,
            format: rt.format,
            width: rt.width,
            height: rt.height,
        }
    }
}

/// Begin a colour-only render pass over `view`. No depth, no multisample.
pub fn begin<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
    load: Load,
) -> wgpu::RenderPass<'a> {
    begin_with(encoder, view, load, wgpu::Color::TRANSPARENT, "fullscreen")
}

/// [`begin`], with a clear colour and a label of your own.
pub fn begin_with<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
    load: Load,
    clear: wgpu::Color,
    label: &str,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: match load {
                    Load::Clear => wgpu::LoadOp::Clear(clear),
                    Load::Keep => wgpu::LoadOp::Load,
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// One fullscreen pass: three vertices, one bind group, no vertex buffer.
pub fn draw(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: Load,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    draw_timed(encoder, view, load, pipeline, bind_group, None);
}

/// [`draw`], with a timestamp written at one end of the pass.
///
/// Hardware without `TIMESTAMP_QUERY_INSIDE_ENCODERS` can only write the
/// GPU's clock at a pass boundary, so timing a stage made of several passes
/// means marking the first pass of each — see
/// [`GpuTimer::mark`](super::gpu_timer::GpuTimer::mark).
pub fn draw_timed(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: Load,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("fullscreen"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: match load {
                    Load::Clear => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    Load::Keep => wgpu::LoadOp::Load,
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: timestamps,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// A linear sampler that clamps at the edge.
///
/// Clamp rather than repeat is the right default for a screen-space pass: a
/// blur that wraps pulls the far edge of the frame into the near one, which
/// reads as a bright fringe along whichever side happens to be busy.
pub fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    sampler(device, wgpu::FilterMode::Linear, "fullscreen linear")
}

/// A nearest sampler that clamps at the edge, for passes that address texels
/// rather than positions.
pub fn nearest_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    sampler(device, wgpu::FilterMode::Nearest, "fullscreen nearest")
}

fn sampler(device: &wgpu::Device, filter: wgpu::FilterMode, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

/// Pipelines that differ only in the format they write, built on first sight.
///
/// A pass that draws into whatever the caller handed it cannot know its target
/// format up front — a surface, an offscreen target and a readback buffer are
/// three different ones — and a pipeline is tied to exactly one. The answer
/// everywhere in this crate is the same small cache, written out three times;
/// this is it once.
pub struct FormatPipelines<T> {
    by_format: HashMap<wgpu::TextureFormat, T>,
}

impl<T> Default for FormatPipelines<T> {
    fn default() -> Self {
        Self {
            by_format: HashMap::new(),
        }
    }
}

impl<T> FormatPipelines<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The entry for `format`, building it with `make` if this is the first
    /// time that format has been seen.
    pub fn get_or_build(&mut self, format: wgpu::TextureFormat, make: impl FnOnce() -> T) -> &T {
        self.by_format.entry(format).or_insert_with(make)
    }

    /// The entry for `format`, if one has been built.
    pub fn get(&self, format: wgpu::TextureFormat) -> Option<&T> {
        self.by_format.get(&format)
    }

    /// How many distinct formats have been built. A number that climbs during
    /// a steady-state frame loop means something is handing over a new format
    /// every frame, which is a leak of pipelines rather than of memory.
    pub fn len(&self) -> usize {
        self.by_format.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_format.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pass in the crate prepends this to its fragment shader, so a
    /// syntax error in it breaks all of them at once. Parsing needs no GPU.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_vertex_stage_is_valid_wgsl() {
        let source = format!(
            "{VS}\n@fragment fn fs(v : FullscreenVertex) -> @location(0) vec4<f32> {{ \
             return vec4<f32>(v.uv, 0.0, 1.0); }}"
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source).expect("the vertex stage parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("the vertex stage validates");
        assert!(VS.contains(&format!("fn {VS_ENTRY}")));
    }

    /// The triangle has to cover the viewport and put uv (0,0) at the top left
    /// — the texture's own origin in WebGPU. Getting this wrong flips every
    /// sampled pass vertically, which still renders and still looks plausible.
    #[test]
    fn the_triangle_covers_the_viewport_with_uv_at_the_top_left() {
        // The vertex stage's arithmetic, evaluated here.
        let vertex = |vid: u32| {
            let x = ((vid << 1) & 2) as f32;
            let y = (vid & 2) as f32;
            ([x, y], [x * 2.0 - 1.0, 1.0 - y * 2.0])
        };
        let vs: Vec<_> = (0..3).map(vertex).collect();
        assert_eq!(vs[0], ([0.0, 0.0], [-1.0, 1.0]), "uv 0,0 is the top left");
        assert_eq!(
            vs[1],
            ([2.0, 0.0], [3.0, 1.0]),
            "second corner is off-screen right"
        );
        assert_eq!(
            vs[2],
            ([0.0, 2.0], [-1.0, -3.0]),
            "third is off-screen below"
        );
        // Every corner of NDC is inside the triangle: barycentric coordinates
        // of each against the three vertices are all non-negative.
        for corner in [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
            let (a, b, c) = (vs[0].1, vs[1].1, vs[2].1);
            let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            let l0 =
                ((b[1] - c[1]) * (corner[0] - c[0]) + (c[0] - b[0]) * (corner[1] - c[1])) / det;
            let l1 =
                ((c[1] - a[1]) * (corner[0] - c[0]) + (a[0] - c[0]) * (corner[1] - c[1])) / det;
            let l2 = 1.0 - l0 - l1;
            assert!(
                l0 >= -1e-6 && l1 >= -1e-6 && l2 >= -1e-6,
                "{corner:?} is outside the triangle"
            );
        }
    }

    #[test]
    fn the_pipeline_cache_builds_once_per_format() {
        let mut built = 0;
        let mut cache: FormatPipelines<u32> = FormatPipelines::new();
        for _ in 0..3 {
            cache.get_or_build(wgpu::TextureFormat::Rgba8Unorm, || {
                built += 1;
                1
            });
        }
        cache.get_or_build(wgpu::TextureFormat::Bgra8Unorm, || {
            built += 1;
            2
        });
        assert_eq!(built, 2, "a format was rebuilt");
        assert_eq!(cache.len(), 2);
    }
}
