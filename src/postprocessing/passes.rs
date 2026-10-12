use crate::math::Color;

/// Sixteen floats, laid out as four `vec4`s, shared by every post pass.
///
/// One fixed block rather than a type per pass, because the composer builds
/// one bind-group layout and every pipeline it compiles uses it. The cost is
/// that a slot's meaning lives in the pass rather than in the type; the saving
/// is that adding a knob to a pass is two lines and touches nothing else.
pub type PassParams = [f32; 16];

/// The declaration the composer prepends to every pass shader, so `params` is
/// in scope. Four `vec4`s named `a` through `d`.
pub(crate) const PASS_PARAMS_WGSL: &str = r#"
struct PassParams {
    a : vec4<f32>,
    b : vec4<f32>,
    c : vec4<f32>,
    d : vec4<f32>,
};
@group(0) @binding(2) var<uniform> params : PassParams;
"#;

/// `Send + Sync` on native, and nothing at all in a browser.
///
/// wgpu's resources are `Arc`-backed natively and `Rc`-backed on wasm, where
/// there are no threads to be safe from. Requiring `Send + Sync` of every
/// pass therefore means *no pass may own a pipeline in a browser* — which is
/// most of what a self-drawing pass is for, and is how [`SsaoPass`] failed to
/// compile for the web before this existed.
#[cfg(not(target_arch = "wasm32"))]
pub trait PassThreading: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> PassThreading for T {}

#[cfg(target_arch = "wasm32")]
pub trait PassThreading {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> PassThreading for T {}

/// Marker trait for a post-process pass.
pub trait Pass: std::fmt::Debug + PassThreading {
    fn name(&self) -> &'static str;
    /// True for the scene-rendering pass (always first).
    fn is_render(&self) -> bool {
        false
    }
    /// WGSL fragment shader source for fullscreen passes that operate on the
    /// previous color buffer (`prev_tex` + `prev_sampler` at @group(0)).
    /// Render passes (`is_render = true`) return None.
    fn shader(&self) -> Option<&'static str> {
        None
    }

    /// The pass's parameters, as the sixteen floats behind `params` in its
    /// shader. `None` for a pass that has none.
    ///
    /// Read fresh every frame, so a value that changes — `FilmPass::time` —
    /// reaches the shader without recompiling anything. The block is a fixed
    /// size because every pass shares one bind-group layout; which slot means
    /// what is up to the pass and its shader, which are written together.
    fn uniforms(&self) -> Option<PassParams> {
        None
    }

    /// True for a pass that draws itself rather than describing a fragment
    /// shader for the composer to run. See [`DrawPass`].
    fn is_draw(&self) -> bool {
        false
    }

    /// Draw, for a pass with [`is_draw`](Self::is_draw). The composer calls
    /// this in chain order with the previous buffer as `input` and the next
    /// one as `target`, and submits the encoder afterwards.
    ///
    /// `&self` rather than `&mut self` because the composer holds its passes
    /// behind a shared borrow while the targets it hands over are borrowed
    /// too; a pass with state of its own keeps it behind a lock, which
    /// [`DrawPass`] does for you.
    fn draw(&self, ctx: &mut DrawContext<'_>) {
        let _ = ctx;
    }
}

/// What a self-drawing pass is given: where the chain has got to, where it
/// should put the result, and the device to do it with.
pub struct DrawContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// Depth from the scene render, for a pass that needs the geometry the
    /// colour buffer no longer carries. `None` if the chain never rendered a
    /// scene.
    pub depth: Option<&'a wgpu::Texture>,
    /// The camera's near and far planes, for turning a depth-buffer value
    /// back into a view-space distance.
    pub near_far: (f32, f32),
    /// `tan(fov_y / 2)` and the viewport aspect, which a pass reconstructing
    /// view-space positions from depth needs and cannot get from the colour
    /// buffer. Taken from the projection matrix, so an orthographic camera
    /// gives the identity-ish values a depth pass will ignore anyway.
    pub tan_half_fov: f32,
    pub aspect: f32,
    /// Record into this; the composer submits it after the pass returns.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// What the chain has produced so far. A pass that ignores it and writes
    /// `target` outright is an overlay; one that samples it is a filter.
    pub input: &'a wgpu::TextureView,
    /// The texture behind `input`, for a pass that needs to copy rather than
    /// sample — a pass that cannot do its job still has to hand the colour on.
    pub input_texture: &'a wgpu::Texture,
    /// Where this pass's output goes, and what the next pass will read.
    pub target: &'a wgpu::TextureView,
    /// The texture behind `target`.
    pub target_texture: &'a wgpu::Texture,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl DrawContext<'_> {
    /// The target as a [`Target`](crate::renderer::fullscreen::Target).
    pub fn target(&self) -> crate::renderer::fullscreen::Target<'_> {
        crate::renderer::fullscreen::Target {
            view: self.target,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }
}

/// A pass that draws itself, from a closure.
///
/// The composer's other passes are *descriptions* — a name and a fragment
/// shader it compiles and runs for you — which covers a filter over the
/// previous buffer and nothing else. Anything that owns GPU state and wants to
/// run inside the chain rather than around it needs this: a fluid simulation,
/// a particle system, a UI layer, a second scene.
///
/// ```no_run
/// # use threers::postprocessing::{DrawPass, EffectComposer, RenderPass};
/// # fn demo(composer: &mut EffectComposer) {
/// composer.add_pass(RenderPass);
/// composer.add_pass(DrawPass::new("fluid", move |ctx| {
///     // record into ctx.encoder, writing ctx.target
///     let _ = &ctx;
/// }));
/// # }
/// ```
///
/// The closure is `FnMut` behind a `Mutex`, because [`Pass`] is `Send + Sync`
/// and a pass that draws has state. The lock is uncontended — the composer
/// calls it from one thread, once a frame.
pub struct DrawPass {
    name: &'static str,
    draw: std::sync::Mutex<DrawFn>,
}

/// The boxed closure, with the threading bound the target can actually meet.
#[cfg(not(target_arch = "wasm32"))]
type DrawFn = Box<dyn FnMut(&mut DrawContext<'_>) + Send>;
#[cfg(target_arch = "wasm32")]
type DrawFn = Box<dyn FnMut(&mut DrawContext<'_>)>;

impl DrawPass {
    pub fn new(
        name: &'static str,
        draw: impl FnMut(&mut DrawContext<'_>) + PassThreading + 'static,
    ) -> Self {
        Self {
            name,
            draw: std::sync::Mutex::new(Box::new(draw)),
        }
    }
}

impl std::fmt::Debug for DrawPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawPass")
            .field("name", &self.name)
            .finish()
    }
}

impl Pass for DrawPass {
    fn name(&self) -> &'static str {
        self.name
    }
    fn is_draw(&self) -> bool {
        true
    }
    fn draw(&self, ctx: &mut DrawContext<'_>) {
        // A panic inside a pass would otherwise poison the lock and turn one
        // bad frame into every later frame failing for a different reason.
        let mut f = match self.draw.lock() {
            Ok(f) => f,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(ctx);
    }
}

/// Renders the scene into the composer's read target. Always the first pass.
#[derive(Debug, Default)]
pub struct RenderPass;
impl Pass for RenderPass {
    fn name(&self) -> &'static str {
        "render"
    }
    fn is_render(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct BloomPass {
    pub strength: f32,
    pub radius: f32,
    pub threshold: f32,
}
impl Default for BloomPass {
    /// The values the shader used to hardcode.
    ///
    /// They are not the numbers this struct used to carry — those were 1.0,
    /// 0.4 and 0.85, and they did nothing, because nothing read them. Making
    /// the fields work and keeping the old defaults would have changed every
    /// picture that uses this pass; these are what it has actually been doing.
    fn default() -> Self {
        Self {
            strength: 0.9,
            radius: 0.02,
            threshold: 0.55,
        }
    }
}
impl Pass for BloomPass {
    fn name(&self) -> &'static str {
        "bloom"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(BLOOM_FRAG)
    }
    fn uniforms(&self) -> Option<PassParams> {
        let mut p = [0.0; 16];
        p[0] = self.strength;
        p[1] = self.radius;
        p[2] = self.threshold;
        Some(p)
    }
}

#[derive(Debug, Default)]
pub struct FxaaPass;
impl Pass for FxaaPass {
    fn name(&self) -> &'static str {
        "fxaa"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(FXAA_FRAG)
    }
}

#[derive(Debug)]
pub struct OutlinePass {
    pub edge_strength: f32,
    pub edge_glow: f32,
    pub edge_thickness: f32,
    pub edge_color: Color,
}
impl Default for OutlinePass {
    fn default() -> Self {
        Self {
            edge_strength: 3.0,
            edge_glow: 0.0,
            edge_thickness: 1.0,
            edge_color: Color::WHITE,
        }
    }
}
impl Pass for OutlinePass {
    fn name(&self) -> &'static str {
        "outline"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(OUTLINE_FRAG)
    }
    fn uniforms(&self) -> Option<PassParams> {
        let mut p = [0.0; 16];
        p[0] = self.edge_strength;
        p[1] = self.edge_thickness;
        p[2] = self.edge_glow;
        p[4] = self.edge_color.r;
        p[5] = self.edge_color.g;
        p[6] = self.edge_color.b;
        Some(p)
    }
}

#[derive(Debug)]
pub struct ToneMappingPass {
    pub exposure: f32,
}
impl Default for ToneMappingPass {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}
impl Pass for ToneMappingPass {
    fn name(&self) -> &'static str {
        "tone_mapping"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(TONE_FRAG)
    }
    fn uniforms(&self) -> Option<PassParams> {
        let mut p = [0.0; 16];
        p[0] = self.exposure;
        Some(p)
    }
}

/// Film grain + scanlines pass. Mirrors three.js's FilmPass.
#[derive(Debug)]
pub struct FilmPass {
    pub grain_intensity: f32,
    pub scanline_intensity: f32,
    pub scanline_count: f32,
    pub time: f32,
}
impl Default for FilmPass {
    /// `grain_intensity` is 0.06, not three.js's 0.5: it is the peak-to-peak
    /// amplitude added to the colour here, and 0.06 is what the shader has
    /// always used. See [`BloomPass::default`] on why the defaults follow the
    /// shader rather than the other way round.
    fn default() -> Self {
        Self {
            grain_intensity: 0.06,
            scanline_intensity: 0.05,
            scanline_count: 480.0,
            time: 0.0,
        }
    }
}
impl Pass for FilmPass {
    fn name(&self) -> &'static str {
        "film"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(FILM_FRAG)
    }
    fn uniforms(&self) -> Option<PassParams> {
        let mut p = [0.0; 16];
        p[0] = self.grain_intensity;
        p[1] = self.scanline_intensity;
        p[2] = self.scanline_count;
        p[3] = self.time;
        Some(p)
    }
}

#[derive(Debug, Default)]
pub struct GlitchPass;
impl Pass for GlitchPass {
    fn name(&self) -> &'static str {
        "glitch"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(GLITCH_FRAG)
    }
}

/// Depth-based ambient occlusion over the scene the chain just rendered.
///
/// This used to return the copy shader — a full-screen copy, costing a pass and
/// doing nothing, with a comment saying the real one lived on the JavaScript
/// side. From Rust you added ambient occlusion and got none, and nothing said
/// so. It now runs [`crate::renderer::ssao`], the same implementation the
/// renderer's own post-fx uses, against the depth buffer the scene render left
/// behind.
pub struct SsaoPass {
    /// How dark the occlusion goes, at 1 being fully black in a crease.
    pub strength: f32,
    /// Sampling radius, **in world units**. It has to suit the scene: a
    /// radius much smaller than the gaps between surfaces finds nothing and
    /// looks like the pass is not running.
    pub radius: f32,
    /// Built on the first frame, and rebuilt when the depth buffer resizes.
    /// Behind a lock because [`Pass`] is `Send + Sync` and this has state.
    state: std::sync::Mutex<Option<SsaoState>>,
}

struct SsaoState {
    ssao: crate::renderer::ssao::Ssao,
    /// The composite that multiplies the colour by the occlusion.
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    size: (u32, u32),
}

impl SsaoPass {
    /// `strength` is how dark a fully occluded crease goes; `radius` is how
    /// far the sampling reaches, in world units.
    ///
    /// A constructor rather than struct literal syntax because the pass
    /// carries the built GPU state, which a caller has no business filling in.
    pub fn new(strength: f32, radius: f32) -> Self {
        Self {
            strength,
            radius,
            state: std::sync::Mutex::new(None),
        }
    }
}

impl Default for SsaoPass {
    /// A radius of one world unit, which suits a scene built at human scale.
    ///
    /// There is no scene-independent answer: the radius is a *distance*, and
    /// at 0.5 on a scene measured in metres the occlusion reaches a hand's
    /// breadth and is nearly invisible, which reads exactly like the pass not
    /// working. Whatever the default, a scene at a different scale has to say
    /// so — which is what the doc comment on the field is for.
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl std::fmt::Debug for SsaoPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SsaoPass")
            .field("strength", &self.strength)
            .field("radius", &self.radius)
            .finish()
    }
}

/// Multiply the colour by the occlusion. Two textures, so it cannot be one of
/// the composer's shader passes, which bind one.
const SSAO_COMPOSITE: &str = r#"
@group(0) @binding(0) var colour : texture_2d<f32>;
@group(0) @binding(1) var occlusion : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;

@fragment
fn fs_main(v : FullscreenVertex) -> @location(0) vec4<f32> {
    let c = textureSample(colour, samp, v.uv);
    // `textureLoad`, not `textureSample`: the occlusion buffer is `R32Float`,
    // which WebGPU does not guarantee is filterable, and binding it as if it
    // were is a validation error rather than a soft fallback. It is at the
    // depth buffer's resolution, so the coordinate is scaled rather than
    // assumed equal.
    let dims = vec2<f32>(textureDimensions(occlusion));
    let at = vec2<i32>(clamp(v.uv * dims, vec2<f32>(0.0), dims - 1.0));
    // Single-channel, 1 where nothing is occluded.
    let ao = textureLoad(occlusion, at, 0).r;
    return vec4<f32>(c.rgb * ao, c.a);
}
"#;

impl Pass for SsaoPass {
    fn name(&self) -> &'static str {
        "ssao"
    }
    fn is_draw(&self) -> bool {
        true
    }
    fn draw(&self, ctx: &mut DrawContext<'_>) {
        let Some(depth) = ctx.depth else {
            // Nothing rendered a scene, so there is no geometry to occlude.
            // Pass the colour through rather than blacking the frame.
            copy_through(ctx);
            return;
        };
        let mut guard = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let size = (depth.width(), depth.height());
        let stale = guard
            .as_ref()
            .is_none_or(|s| s.size != size || s.format != ctx.format);
        if stale {
            let (near, far) = ctx.near_far;
            let Some(ssao) = crate::renderer::ssao::Ssao::new(
                ctx.device,
                depth,
                near,
                far,
                self.radius,
                self.strength,
                ctx.tan_half_fov,
                ctx.aspect,
            ) else {
                // No compute *and* no raster path means this device cannot run
                // it at all, which is worth one line rather than silence.
                log::warn!("SsaoPass: this device cannot build the ambient-occlusion pass");
                copy_through(ctx);
                return;
            };
            *guard = Some(SsaoState::new(ctx, ssao, size));
        }
        let state = guard.as_ref().expect("just built");
        state.ssao.record(ctx.encoder);

        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssao composite"),
            layout: &state.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ctx.input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(state.ssao.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&state.sampler),
                },
            ],
        });
        crate::renderer::fullscreen::draw(
            ctx.encoder,
            ctx.target,
            crate::renderer::fullscreen::Load::Clear,
            &state.pipeline,
            &bind,
        );
    }
}

impl SsaoState {
    fn new(ctx: &DrawContext<'_>, ssao: crate::renderer::ssao::Ssao, size: (u32, u32)) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ssao composite"),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{}\n{SSAO_COMPOSITE}", crate::renderer::fullscreen::VS).into(),
                ),
            });
        let tex = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao composite bgl"),
                entries: &[
                    tex(0, true),
                    // The occlusion buffer is `R32Float` and loaded, not
                    // sampled.
                    tex(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ssao composite layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ssao composite"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(crate::renderer::fullscreen::VS_ENTRY),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ctx.format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        Self {
            ssao,
            pipeline,
            layout,
            sampler: crate::renderer::fullscreen::linear_sampler(ctx.device),
            format: ctx.format,
            size,
        }
    }
}

/// Hand the colour on untouched — what a draw pass does when it cannot do its
/// job, since writing nothing would leave the next pass reading a cleared
/// buffer.
fn copy_through(ctx: &mut DrawContext<'_>) {
    ctx.encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: ctx.input_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: ctx.target_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: ctx.width,
            height: ctx.height,
            depth_or_array_layers: 1,
        },
    );
}

/// Screen-space reflections — **not implemented in the Rust composer.**
///
/// Like [`SsaoPass`] this used to be a silent full-screen copy. Unlike it, the
/// honest fix is not to implement it here: screen-space reflections need
/// surface normals, the composer's chain carries colour and depth and no
/// normals, and normals reconstructed from depth derivatives give a reflection
/// that is wrong along every silhouette — which is precisely where a
/// reflection is looked at. A bad one is worse than none.
///
/// So this declines: [`shader`](Pass::shader) returns `None`, the composer
/// skips it entirely rather than spending a pass copying, and adding it logs
/// once. The working implementation is the renderer's own post-fx
/// (`POSTFX_SHADER` kind 12), which has the normal prepass; the JavaScript
/// `SSRPass` drives that.
#[derive(Debug, Default)]
pub struct SsrPass;
impl Pass for SsrPass {
    fn name(&self) -> &'static str {
        "ssr"
    }
    fn shader(&self) -> Option<&'static str> {
        log::warn!(
            "SsrPass does nothing in the Rust EffectComposer: screen-space \
             reflections need normals, which this chain does not carry. Use \
             the renderer's own post-fx path."
        );
        None
    }
}

#[derive(Debug, Default)]
pub struct CopyPass;
impl Pass for CopyPass {
    fn name(&self) -> &'static str {
        "copy"
    }
    fn shader(&self) -> Option<&'static str> {
        Some(COPY_FRAG)
    }
}

// ---- WGSL fragment shaders ----
// All passes share a common vertex shader (full-screen triangle) provided by
// EffectComposer.

pub(crate) const COPY_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(prev_tex, prev_sampler, uv);
}
"#;

pub(crate) const FXAA_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
fn rgb_luma(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(prev_tex));
    let inv = vec2<f32>(1.0) / dims;
    let c   = textureSample(prev_tex, prev_sampler, uv);
    let nw  = textureSample(prev_tex, prev_sampler, uv + vec2<f32>(-inv.x, -inv.y)).rgb;
    let ne  = textureSample(prev_tex, prev_sampler, uv + vec2<f32>( inv.x, -inv.y)).rgb;
    let sw  = textureSample(prev_tex, prev_sampler, uv + vec2<f32>(-inv.x,  inv.y)).rgb;
    let se  = textureSample(prev_tex, prev_sampler, uv + vec2<f32>( inv.x,  inv.y)).rgb;
    let l_c  = rgb_luma(c.rgb);
    let l_nw = rgb_luma(nw);
    let l_ne = rgb_luma(ne);
    let l_sw = rgb_luma(sw);
    let l_se = rgb_luma(se);
    let l_min = min(l_c, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_c, max(max(l_nw, l_ne), max(l_sw, l_se)));
    if (l_max - l_min < 0.05) {
        return c;
    }
    var dir = vec2<f32>(
        -((l_nw + l_ne) - (l_sw + l_se)),
         ((l_nw + l_sw) - (l_ne + l_se)),
    );
    let rcp_dir = 1.0 / (min(abs(dir.x), abs(dir.y)) + 0.00078125);
    dir = clamp(dir * rcp_dir, vec2<f32>(-8.0), vec2<f32>(8.0)) * inv;
    let s1 = textureSample(prev_tex, prev_sampler, uv + dir * (1.0 / 3.0 - 0.5)).rgb;
    let s2 = textureSample(prev_tex, prev_sampler, uv + dir * (2.0 / 3.0 - 0.5)).rgb;
    let blend = (s1 + s2) * 0.5;
    return vec4<f32>(blend, c.a);
}
"#;

pub(crate) const BLOOM_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
fn luma(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }
// Jimenez interleaved-gradient noise → a per-pixel angle in [0, 1).
fn ign(frag : vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715))));
}
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let c = textureSample(prev_tex, prev_sampler, uv);
    let dims = vec2<f32>(textureDimensions(prev_tex));
    let px = vec2<f32>(1.0) / dims;
    let min_dim = min(dims.x, dims.y);
    // Halo radius as a fraction of the image → resolution-independent, wide.
    let radius = min_dim * params.a.y;
    let threshold = params.a.z;

    // A regular grid of wide-spaced taps replicates a thin bright arc into a
    // lattice of concentric ghost copies (the "ripples"). Sample a golden-angle
    // spiral instead — no aligned grid, no rings — and rotate the whole spiral
    // by a per-pixel angle so any residual structure dithers into fine noise
    // that reads as a smooth glow rather than banded rings.
    let TAPS = 128;
    let GOLDEN = 2.3999632;                 // golden angle (radians)
    let base = ign(uv * dims) * 6.2831853;  // per-pixel spiral rotation
    var sum = vec3<f32>(0.0);
    var wsum = 1e-6;
    for (var i: i32 = 0; i < TAPS; i = i + 1) {
        let fi = f32(i) + 0.5;
        let rn = sqrt(fi / f32(TAPS));       // even areal density across the disk
        let ang = fi * GOLDEN + base;
        let off = vec2<f32>(cos(ang), sin(ang)) * (rn * radius) * px;
        let s = textureSampleLevel(prev_tex, prev_sampler, uv + off, 0.0).rgb;
        // Keep the full color of bright pixels (soft threshold), not just the
        // sliver above the threshold — otherwise the bloom is near-invisible.
        let mask = smoothstep(threshold, threshold + 0.25, luma(s));
        let w = exp(-rn * rn * 3.0);         // Gaussian falloff toward the rim
        sum = sum + s * mask * w;
        wsum = wsum + w;
    }
    let bloom = sum / wsum;
    return vec4<f32>(c.rgb + bloom * params.a.x, c.a);
}
"#;

pub(crate) const OUTLINE_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
fn luma(c : vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let strength  = params.a.x;
    let thickness = params.a.y;
    let glow      = params.a.z;
    let tint      = params.b.rgb;

    let dims = vec2<f32>(textureDimensions(prev_tex));
    let inv  = (vec2<f32>(1.0) / dims) * thickness;
    let lx0  = luma(textureSample(prev_tex, prev_sampler, uv + vec2<f32>(-inv.x, 0.0)).rgb);
    let lx1  = luma(textureSample(prev_tex, prev_sampler, uv + vec2<f32>( inv.x, 0.0)).rgb);
    let ly0  = luma(textureSample(prev_tex, prev_sampler, uv + vec2<f32>(0.0, -inv.y)).rgb);
    let ly1  = luma(textureSample(prev_tex, prev_sampler, uv + vec2<f32>(0.0,  inv.y)).rgb);
    let edge = abs(lx0 - lx1) + abs(ly0 - ly1);
    let c    = textureSample(prev_tex, prev_sampler, uv).rgb;
    let line = clamp(edge * strength, 0.0, 1.0);
    // `glow` adds the edge on top of the line rather than blending into it,
    // so a lit outline reads as emission instead of as a wider line.
    let out  = mix(c, tint, line) + tint * (edge * glow);
    return vec4<f32>(out, 1.0);
}
"#;

pub(crate) const TONE_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
fn aces(x : vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let c = textureSample(prev_tex, prev_sampler, uv);
    return vec4<f32>(aces(c.rgb * params.a.x), c.a);
}
"#;

pub(crate) const FILM_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
fn rand(co : vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let grain_intensity    = params.a.x;
    let scanline_intensity = params.a.y;
    let scanline_count     = params.a.z;
    let time               = params.a.w;

    let c = textureSample(prev_tex, prev_sampler, uv);
    let scan = (1.0 - scanline_intensity) + scanline_intensity * sin(uv.y * scanline_count * 3.14159);
    // `time` moves the noise field, so grain crawls between frames instead of
    // sitting still — which is what makes it read as film rather than as dirt
    // on the lens.
    let grain = (rand(uv * 1024.0 + vec2<f32>(time, time * 1.37)) - 0.5) * grain_intensity;
    return vec4<f32>(c.rgb * scan + vec3<f32>(grain), c.a);
}
"#;

pub(crate) const GLITCH_FRAG: &str = r#"
@group(0) @binding(0) var prev_tex : texture_2d<f32>;
@group(0) @binding(1) var prev_sampler : sampler;
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let band = step(0.97, fract(uv.y * 21.7));
    let shifted = textureSample(prev_tex, prev_sampler, uv + vec2<f32>(band * 0.04, 0.0));
    return shifted;
}
"#;
